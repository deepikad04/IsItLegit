import asyncio
import json
import secrets
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload
from cachetools import TTLCache

from database import get_db
from models.user import User
from models.scenario import Scenario
from models.simulation import Simulation
from models.decision import Decision
from models.simulation_snapshot import SimulationSnapshot
from routers.auth import get_current_user, limiter
from services.simulation_engine import SimulationEngine
from services.gemini_service import GeminiService
from schemas.simulation import (
    SimulationCreate,
    SimulationResponse,
    SimulationState,
    SimulationComplete,
    SimulationOutcome
)
from schemas.decision import DecisionCreate, DecisionResponse

router = APIRouter(prefix="/api/simulations", tags=["simulations"])

# ── Stream token cache: opaque token → {simulation_id, user_id, expires} ──
_stream_tokens: TTLCache = TTLCache(maxsize=512, ttl=60)


@router.post("/start", response_model=SimulationState)
async def start_simulation(
    data: SimulationCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db)
):
    """Start a new simulation session."""
    # Get scenario
    scenario = db.query(Scenario).filter(Scenario.id == data.scenario_id).first()
    if not scenario:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scenario not found"
        )

    # Check for existing in-progress simulation
    existing = db.query(Simulation).filter(
        Simulation.user_id == current_user.id,
        Simulation.status == "in_progress"
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You have an active simulation. Complete or abandon it first."
        )

    # Create simulation
    engine = SimulationEngine(scenario)
    initial_state = engine.get_initial_state()

    simulation = Simulation(
        user_id=current_user.id,
        scenario_id=scenario.id,
        status="in_progress",
        current_portfolio=initial_state["portfolio"],
        current_time_elapsed=0
    )
    db.add(simulation)
    db.commit()
    db.refresh(simulation)

    return SimulationState(
        id=simulation.id,
        scenario_id=scenario.id,
        scenario_name=scenario.name,
        status="in_progress",
        time_elapsed=0,
        time_remaining=scenario.time_pressure_seconds,
        current_price=initial_state["current_price"],
        price_history=initial_state["price_history"],
        portfolio=initial_state["portfolio"],
        available_info=initial_state["available_info"],
        recent_events=[]
    )


@router.get("/{simulation_id}/state", response_model=SimulationState)
async def get_simulation_state(
    simulation_id: UUID,
    time_elapsed: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db)
):
    """Get current simulation state at a given time."""
    simulation = db.query(Simulation).filter(
        Simulation.id == simulation_id,
        Simulation.user_id == current_user.id
    ).first()

    if not simulation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Simulation not found"
        )

    if simulation.status != "in_progress":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Simulation is not active"
        )

    scenario = simulation.scenario
    engine = SimulationEngine(scenario)
    state = engine.get_state_at_time(time_elapsed, simulation.current_portfolio)

    # Buffer DB writes: only persist every 15s or on significant time jumps
    if time_elapsed % 15 == 0 or time_elapsed == 0:
        simulation.current_time_elapsed = time_elapsed
        db.commit()

    return SimulationState(
        id=simulation.id,
        scenario_id=scenario.id,
        scenario_name=scenario.name,
        status="in_progress",
        time_elapsed=time_elapsed,
        time_remaining=max(0, scenario.time_pressure_seconds - time_elapsed),
        current_price=state["current_price"],
        price_history=state["price_history"],
        portfolio=state["portfolio"],
        available_info=state["available_info"],
        recent_events=state["recent_events"],
        market_conditions=state.get("market_conditions"),
        last_execution=state.get("last_execution"),
        pending_orders=state.get("pending_orders"),
    )


@router.post("/{simulation_id}/decision", response_model=DecisionResponse)
async def make_decision(
    simulation_id: UUID,
    decision_data: DecisionCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db)
):
    """Log a decision in the simulation."""
    simulation = db.query(Simulation).filter(
        Simulation.id == simulation_id,
        Simulation.user_id == current_user.id
    ).first()

    if not simulation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Simulation not found"
        )

    if simulation.status != "in_progress":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Simulation is not active"
        )

    scenario = simulation.scenario
    engine = SimulationEngine(scenario)

    # Use client-reported time if provided and within valid range of DB value
    effective_time = simulation.current_time_elapsed
    if decision_data.time_elapsed is not None:
        client_time = decision_data.time_elapsed
        # Accept if within a reasonable window (client can be up to 15s ahead of last DB write)
        if 0 <= client_time <= scenario.time_pressure_seconds:
            effective_time = client_time
            # Sync DB to client time
            simulation.current_time_elapsed = effective_time

    current_state = engine.get_state_at_time(
        effective_time,
        simulation.current_portfolio
    )

    # Process the decision (pass time_elapsed for halt/spread/time-pressure checks)
    new_portfolio = engine.process_decision(
        decision_data,
        current_state,
        simulation.current_portfolio,
        time_elapsed=effective_time
    )

    # Get events since last decision
    last_decision = db.query(Decision).filter(
        Decision.simulation_id == simulation_id
    ).order_by(Decision.simulation_time.desc()).first()

    last_time = last_decision.simulation_time if last_decision else 0
    events_since = engine.get_events_between(last_time, effective_time)

    # Store snapshot first (Phase 2.1: deduplication)
    snapshot = SimulationSnapshot(
        simulation_id=simulation_id,
        simulation_time=effective_time,
        snapshot_type="decision_context",
        data=current_state,
    )
    db.add(snapshot)
    db.flush()  # get snapshot.id

    # Create decision record with snapshot_id reference instead of full state blob
    decision = Decision(
        simulation_id=simulation_id,
        simulation_time=effective_time,
        decision_type=decision_data.decision_type,
        asset=decision_data.asset or scenario.initial_data.get("asset"),
        amount=decision_data.amount,
        confidence_level=decision_data.confidence_level,
        time_spent_seconds=decision_data.time_spent_seconds,
        rationale=decision_data.rationale,
        price_at_decision=current_state["current_price"],
        info_viewed=[iv.model_dump() for iv in (decision_data.info_viewed or [])],
        info_ignored=current_state.get("info_available_but_not_viewed", []),
        snapshot_id=snapshot.id,
        market_state_at_decision=None,  # No longer duplicating state blob
        events_since_last=events_since
    )
    db.add(decision)

    # Update simulation portfolio
    simulation.current_portfolio = new_portfolio
    db.commit()
    db.refresh(decision)

    return decision


# ── Decision Challenge Mode (Phase 2.4) ──────────────────────────────

class ChallengeRequest(BaseModel):
    decision_type: str
    amount: float | None = None
    rationale: str


@router.post("/{simulation_id}/challenge")
async def challenge_decision(
    simulation_id: UUID,
    data: ChallengeRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db)
):
    """AI rates the user's reasoning BEFORE confirming a decision."""
    simulation = db.query(Simulation).filter(
        Simulation.id == simulation_id,
        Simulation.user_id == current_user.id
    ).first()

    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation not found")
    if simulation.status != "in_progress":
        raise HTTPException(status_code=400, detail="Simulation is not active")

    scenario = simulation.scenario
    engine = SimulationEngine(scenario)
    current_state = engine.get_state_at_time(
        simulation.current_time_elapsed,
        simulation.current_portfolio
    )

    decisions_so_far = db.query(Decision).filter(
        Decision.simulation_id == simulation_id
    ).order_by(Decision.simulation_time).all()

    gemini = GeminiService()
    result = await gemini.challenge_reasoning(
        decision_type=data.decision_type,
        amount=data.amount,
        rationale=data.rationale,
        scenario=scenario,
        current_state=current_state,
        decisions_so_far=decisions_so_far,
    )

    return result


class SkipTimeRequest(BaseModel):
    seconds: int


@router.post("/{simulation_id}/skip-time", response_model=SimulationState)
async def skip_time(
    simulation_id: UUID,
    data: SkipTimeRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """Fast-forward simulation time by a given number of seconds."""
    simulation = db.query(Simulation).filter(
        Simulation.id == simulation_id,
        Simulation.user_id == current_user.id,
    ).first()

    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation not found")
    if simulation.status != "in_progress":
        raise HTTPException(status_code=400, detail="Simulation is not active")

    scenario = simulation.scenario
    seconds = max(1, min(data.seconds, 120))
    new_time = min(
        simulation.current_time_elapsed + seconds,
        scenario.time_pressure_seconds,
    )

    engine = SimulationEngine(scenario)
    state = engine.get_state_at_time(new_time, simulation.current_portfolio)

    simulation.current_time_elapsed = new_time
    db.commit()

    return SimulationState(
        id=simulation.id,
        scenario_id=scenario.id,
        scenario_name=scenario.name,
        status="in_progress",
        time_elapsed=new_time,
        time_remaining=max(0, scenario.time_pressure_seconds - new_time),
        current_price=state["current_price"],
        price_history=state["price_history"],
        portfolio=state["portfolio"],
        available_info=state["available_info"],
        recent_events=state["recent_events"],
        market_conditions=state.get("market_conditions"),
        pending_orders=state.get("pending_orders"),
    )


@router.post("/{simulation_id}/complete", response_model=SimulationOutcome)
async def complete_simulation(
    simulation_id: UUID,
    complete_data: SimulationComplete,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db)
):
    """Complete a simulation and get results."""
    simulation = db.query(Simulation).filter(
        Simulation.id == simulation_id,
        Simulation.user_id == current_user.id
    ).first()

    if not simulation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Simulation not found"
        )

    if simulation.status != "in_progress":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Simulation is not active"
        )

    scenario = simulation.scenario
    engine = SimulationEngine(scenario)

    # Calculate final outcome
    final_state = engine.get_final_state(simulation.current_portfolio)
    decisions = db.query(Decision).filter(
        Decision.simulation_id == simulation_id
    ).all()

    # Calculate process quality (includes market realism scoring factors)
    process_score = engine.calculate_process_quality(decisions, scenario, portfolio=simulation.current_portfolio)

    # Calculate profit/loss
    initial_value = scenario.initial_data.get("your_balance", 10000)
    final_value = final_state["total_value"]
    profit_loss = final_value - initial_value
    profit_loss_percent = (profit_loss / initial_value) * 100

    # Precompute summary fields
    outcome_type = "profit" if profit_loss > 100 else ("loss" if profit_loss < -100 else "break_even")
    outcome_summary = f"+${profit_loss:.2f}" if profit_loss >= 0 else f"-${abs(profit_loss):.2f}"

    # Update simulation
    simulation.status = "completed"
    simulation.completed_at = datetime.utcnow()
    simulation.final_outcome = {
        "final_value": final_value,
        "profit_loss": profit_loss,
        "profit_loss_percent": profit_loss_percent,
        "final_portfolio": final_state,
        "process_quality_score": process_score,
        "outcome_type": outcome_type,
        "outcome_summary": outcome_summary,
        "total_decisions": len(decisions),
        "time_taken": simulation.current_time_elapsed,
    }
    simulation.process_quality_score = process_score

    # Store completion snapshot
    completion_snapshot = SimulationSnapshot(
        simulation_id=simulation_id,
        simulation_time=simulation.current_time_elapsed,
        snapshot_type="completion",
        data=final_state,
    )
    db.add(completion_snapshot)

    # Update user stats
    current_user.total_simulations += 1
    current_user.last_simulation_date = datetime.utcnow()

    db.commit()

    return SimulationOutcome(
        simulation_id=simulation.id,
        profit_loss=profit_loss,
        profit_loss_percent=profit_loss_percent,
        final_portfolio_value=final_value,
        process_quality_score=process_score,
        total_decisions=len(decisions),
        time_taken=simulation.current_time_elapsed,
        outcome_type=outcome_type,
        outcome_summary=outcome_summary,
        total_fees_paid=simulation.current_portfolio.get("cumulative_fees", 0),
    )


@router.post("/{simulation_id}/abandon")
async def abandon_simulation(
    simulation_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db)
):
    """Abandon an in-progress simulation."""
    simulation = db.query(Simulation).filter(
        Simulation.id == simulation_id,
        Simulation.user_id == current_user.id
    ).first()

    if not simulation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Simulation not found"
        )

    simulation.status = "abandoned"
    simulation.completed_at = datetime.utcnow()
    db.commit()

    return {"message": "Simulation abandoned"}


@router.get("/", response_model=list[SimulationResponse])
async def list_simulations(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
    status: str | None = None,
    limit: int = 20
):
    """List user's simulations."""
    query = (
        db.query(Simulation)
        .options(joinedload(Simulation.scenario))
        .filter(Simulation.user_id == current_user.id)
    )

    if status:
        query = query.filter(Simulation.status == status)

    return query.order_by(Simulation.started_at.desc()).limit(limit).all()


# ── Credibility Check (Google Search Grounding) ─────────────────────

@router.post("/verify-credibility")
@limiter.limit("5/minute")
async def verify_credibility(
    request: Request,
    current_user: Annotated[User, Depends(get_current_user)],
    claim: str = "",
    source_type: str = "news",
):
    """Verify a news claim or social post using Google Search grounding."""
    if not claim:
        raise HTTPException(status_code=400, detail="Claim text is required")

    gemini = GeminiService()
    return await gemini.verify_claim_credibility(claim, source_type)


# ── Stream Token (Phase 1.2: hardened SSE auth) ─────────────────────

@router.post("/{simulation_id}/stream-token")
async def get_stream_token(
    simulation_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """Issue a short-lived opaque token for SSE streaming."""
    simulation = db.query(Simulation).filter(
        Simulation.id == simulation_id,
        Simulation.user_id == current_user.id,
    ).first()
    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation not found")
    if simulation.status != "in_progress":
        raise HTTPException(status_code=400, detail="Simulation is not active")

    token = secrets.token_urlsafe(32)
    _stream_tokens[token] = {
        "simulation_id": str(simulation_id),
        "user_id": str(current_user.id),
    }
    return {"token": token}


@router.get("/{simulation_id}/stream")
async def stream_simulation(
    simulation_id: UUID,
    token: str = Query(..., description="Stream token for SSE auth"),
    db: Session = Depends(get_db),
):
    """SSE endpoint for real-time simulation state updates with live coaching."""
    from database import SessionLocal

    # Validate opaque stream token (issued via POST /stream-token)
    token_data = _stream_tokens.get(token)
    if not token_data:
        raise HTTPException(status_code=401, detail="Invalid or expired stream token")

    if token_data["simulation_id"] != str(simulation_id):
        raise HTTPException(status_code=403, detail="Token does not match simulation")

    simulation = db.query(Simulation).filter(
        Simulation.id == simulation_id,
        Simulation.user_id == UUID(token_data["user_id"]),
    ).first()

    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation not found")
    if simulation.status != "in_progress":
        raise HTTPException(status_code=400, detail="Simulation is not active")

    # Capture immutable data before entering generator (db session closes after response setup)
    scenario = simulation.scenario
    # Eagerly load all scenario attributes we need, then detach from session
    # so the generator doesn't trip on a closed session accessing lazy attrs
    _ = scenario.id, scenario.name, scenario.initial_data, scenario.events
    _ = scenario.time_pressure_seconds, scenario.category, scenario.difficulty
    db.expunge(scenario)

    engine = SimulationEngine(scenario)
    initial_time = simulation.current_time_elapsed
    initial_portfolio = simulation.current_portfolio

    async def event_generator():
        time_elapsed = initial_time
        current_portfolio = initial_portfolio
        last_decision_count = 0
        last_nudge_time = -30  # cooldown tracker

        while time_elapsed <= scenario.time_pressure_seconds:
            # Open a short-lived session per tick to avoid pool exhaustion
            tick_db = SessionLocal()
            try:
                sim = tick_db.query(Simulation).filter(
                    Simulation.id == simulation_id
                ).first()
                if not sim or sim.status != "in_progress":
                    yield f"data: {json.dumps({'status': sim.status if sim else 'abandoned'})}\n\n"
                    break

                current_portfolio = sim.current_portfolio

                state = engine.get_state_at_time(time_elapsed, current_portfolio)

                # Check and execute pending limit/stop orders each tick
                portfolio = state["portfolio"]
                if portfolio.get("pending_orders"):
                    filled = engine._check_pending_orders(portfolio, state["current_price"], time_elapsed)
                    if filled:
                        sim.current_portfolio = portfolio
                        current_portfolio = portfolio
                        tick_db.commit()

                payload = {
                    "time_elapsed": time_elapsed,
                    "time_remaining": max(0, scenario.time_pressure_seconds - time_elapsed),
                    "current_price": state["current_price"],
                    "price_history": state["price_history"],
                    "portfolio": state["portfolio"],
                    "available_info": state["available_info"],
                    "recent_events": state["recent_events"],
                    "market_conditions": state.get("market_conditions"),
                    "pending_orders": state.get("pending_orders"),
                    "status": "in_progress",
                }
                yield f"data: {json.dumps(payload, default=str)}\n\n"

                # Buffer DB writes — only persist every 15s or at boundaries
                if time_elapsed % 15 == 0 or time_elapsed == 0:
                    sim.current_time_elapsed = time_elapsed
                    tick_db.commit()

                # Live Coach Interventions
                decision_count = tick_db.query(Decision).filter(
                    Decision.simulation_id == simulation_id
                ).count()

                if (decision_count > last_decision_count
                        and time_elapsed - last_nudge_time >= 30):
                    last_decision_count = decision_count
                    recent_decisions = tick_db.query(Decision).filter(
                        Decision.simulation_id == simulation_id
                    ).order_by(Decision.simulation_time.desc()).limit(3).all()
                    recent_decisions.reverse()
                    try:
                        gemini = GeminiService()
                        nudge = await gemini.generate_live_nudge(
                            recent_decisions, scenario, time_elapsed
                        )
                        if nudge:
                            yield f"event: coach\ndata: {json.dumps(nudge)}\n\n"
                            last_nudge_time = time_elapsed
                    except Exception:
                        pass  # Never break the stream for coach failures
            finally:
                tick_db.close()

            await asyncio.sleep(1)
            time_elapsed += 1

        # Final DB sync
        final_db = SessionLocal()
        try:
            sim = final_db.query(Simulation).filter(Simulation.id == simulation_id).first()
            if sim:
                sim.current_time_elapsed = time_elapsed
                final_db.commit()
        finally:
            final_db.close()

        yield f"event: complete\ndata: {json.dumps({'status': 'time_expired'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
