#!/usr/bin/env python3
"""
MCP Server for Airline Tools using FastMCP with SSE (Server-Sent Events)

This server exposes all airline tools from tau-bench as MCP tools with proper JSON schemas.
Run with: python airline_tools_mcp_fixed.py
MCP URL: http://localhost:8000/sse
Reload endpoint: POST http://localhost:8001/reload
"""

import logging
from typing import Any, Dict, List, Annotated
import threading
import asyncio

from pydantic import BaseModel, Field
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse
from starlette.requests import Request
import uvicorn

from tau_bench.envs.airline.data import load_data
from tau_bench.envs.airline.tools import (
    BookReservation,
    Calculate,
    CancelReservation,
    GetReservationDetails,
    GetUserDetails,
    ListAllAirports,
    SearchDirectFlight,
    SearchOnestopFlight,
    SendCertificate,
    Think,
    TransferToHumanAgents,
    UpdateReservationBaggages,
    UpdateReservationFlights,
    UpdateReservationPassengers,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("airline-tools-mcp")

# Create FastMCP server
mcp = FastMCP("Airline Tools")

# Load airline data globally (mutable dict)
logger.info("Loading airline data...")
airline_data = load_data()
logger.info(f"Loaded {len(airline_data.get('flights', {}))} flights, "
            f"{len(airline_data.get('reservations', {}))} reservations, "
            f"{len(airline_data.get('users', {}))} users")


# Define Pydantic models for nested structures to preserve all fields
class FlightInfo(BaseModel):
    flight_number: str = Field(description="Flight number, such as 'HAT001'.")
    date: str = Field(description="The date for the flight in the format 'YYYY-MM-DD', such as '2024-05-01'.")

class PassengerInfo(BaseModel):
    first_name: str = Field(description="The first name of the passenger, such as 'Noah'.")
    last_name: str = Field(description="The last name of the passenger, such as 'Brown'.")
    dob: str = Field(description="The date of birth of the passenger in the format 'YYYY-MM-DD', such as '1990-01-01'.")

class PaymentMethod(BaseModel):
    payment_id: str = Field(description="The payment id stored in user profile, such as 'credit_card_7815826', 'gift_card_7815826', 'certificate_7815826'.")
    amount: float = Field(description="The amount to be paid.")


def reload_database():
    """Reload the airline database from source files."""
    global airline_data
    logger.info("Reloading airline database...")
    new_data = load_data()
    airline_data.clear()
    airline_data.update(new_data)
    logger.info(f"Database reloaded: {len(airline_data.get('flights', {}))} flights, "
                f"{len(airline_data.get('reservations', {}))} reservations, "
                f"{len(airline_data.get('users', {}))} users")
    return airline_data


async def reload_endpoint(request: Request):
    """REST endpoint to reload the database."""
    reload_database()
    return JSONResponse({
        "status": "success",
        "message": "Database reloaded",
        "stats": {
            "flights": len(airline_data.get('flights', {})),
            "reservations": len(airline_data.get('reservations', {})),
            "users": len(airline_data.get('users', {}))
        }
    })


# Define Annotated types with descriptions from tau-bench schemas
UserId = Annotated[str, "The ID of the user to book the reservation, such as 'sara_doe_496'."]
OriginIATA = Annotated[str, "The IATA code for the origin city, such as 'SFO'."]
DestinationIATA = Annotated[str, "The IATA code for the destination city, such as 'JFK'."]
FlightType = Annotated[str, "The flight type: 'one_way' or 'round_trip'."]
Cabin = Annotated[str, "The cabin class: 'basic_economy', 'economy', or 'business'."]
# Use Pydantic models for complex nested structures
FlightsList = Annotated[List[FlightInfo], "An array of flight objects with flight_number and date."]
PassengersList = Annotated[List[PassengerInfo], "An array of passenger objects with first_name, last_name, and dob."]
PaymentMethodsList = Annotated[List[PaymentMethod], "An array of payment method objects with payment_id and amount."]
TotalBaggages = Annotated[int, "The total number of baggage items included in the reservation."]
NonfreeBaggages = Annotated[int, "The number of non-free baggage items included in the reservation."]
Insurance = Annotated[str, "Whether to include travel insurance: 'yes' or 'no'."]
ReservationId = Annotated[str, "The reservation ID."]
FlightDate = Annotated[str, "The date of the flight in the format 'YYYY-MM-DD', such as '2024-01-01'."]
CalculationOperation = Annotated[str, "The calculation operation to perform, e.g., '2 + 2'."]
Thought = Annotated[str, "Internal reasoning or thought process."]
Summary = Annotated[str, "Summary of the conversation for human agents."]
CertificateAmount = Annotated[int, "The amount of the certificate to send to the user."]


@mcp.tool(description="Book a reservation for a user with specified flights, passengers, and payment methods.")
def book_reservation(
    user_id: UserId,
    origin: OriginIATA,
    destination: DestinationIATA,
    flight_type: FlightType,
    cabin: Cabin,
    flights: FlightsList,
    passengers: PassengersList,
    payment_methods: PaymentMethodsList,
    total_baggages: TotalBaggages,
    nonfree_baggages: NonfreeBaggages,
    insurance: Insurance,
) -> str:
    """Book a reservation."""
    # Convert Pydantic models to dicts for tau-bench
    flights_dicts = [f.model_dump() for f in flights]
    passengers_dicts = [p.model_dump() for p in passengers]
    payment_methods_dicts = [pm.model_dump() for pm in payment_methods]

    return BookReservation.invoke(
        airline_data,
        user_id,
        origin,
        destination,
        flight_type,
        cabin,
        flights_dicts,
        passengers_dicts,
        payment_methods_dicts,
        total_baggages,
        nonfree_baggages,
        insurance,
    )


@mcp.tool(description="Perform a calculation operation.")
def calculate(operation: CalculationOperation) -> str:
    """Perform a calculation."""
    return Calculate.invoke(airline_data, operation)


@mcp.tool(description="Cancel a reservation by its ID.")
def cancel_reservation(reservation_id: ReservationId) -> str:
    """Cancel a reservation."""
    return CancelReservation.invoke(airline_data, reservation_id)


@mcp.tool(description="Get the details of a reservation by its ID.")
def get_reservation_details(reservation_id: ReservationId) -> str:
    """Get the details of a reservation."""
    return GetReservationDetails.invoke(airline_data, reservation_id)


@mcp.tool(description="Get the details of a user, including their reservations.")
def get_user_details(user_id: UserId) -> str:
    """Get the details of a user, including their reservations."""
    return GetUserDetails.invoke(airline_data, user_id)


@mcp.tool(description="List all available airports with their IATA codes and city names. Returns a JSON object mapping IATA codes to city names (e.g., {'JFK': 'New York', 'LAX': 'Los Angeles'}).")
def list_all_airports() -> str:
    """List all airports."""
    return ListAllAirports.invoke(airline_data)


@mcp.tool(description="Search for direct flights between two cities on a specific date. Use three-letter IATA airport codes.")
def search_direct_flight(
    origin: Annotated[str, "The origin city airport in three letters, such as 'JFK'."],
    destination: Annotated[str, "The destination city airport in three letters, such as 'LAX'."],
    date: FlightDate
) -> str:
    """Search direct flights between two cities on a specific date."""
    return SearchDirectFlight.invoke(airline_data, origin, destination, date)


@mcp.tool(description="Search for one-stop flights between two cities on a specific date. Use three-letter IATA airport codes.")
def search_onestop_flight(
    origin: Annotated[str, "The origin city airport in three letters, such as 'JFK'."],
    destination: Annotated[str, "The destination city airport in three letters, such as 'LAX'."],
    date: FlightDate
) -> str:
    """Search one-stop flights between two cities on a specific date."""
    return SearchOnestopFlight.invoke(airline_data, origin, destination, date)


@mcp.tool(description="Send a certificate (voucher/credit) to a user.")
def send_certificate(user_id: UserId, amount: CertificateAmount) -> str:
    """Send a certificate to a user."""
    return SendCertificate.invoke(airline_data, user_id, amount)


@mcp.tool(description="Think about something (internal reasoning). Use this to plan your approach or reason through a problem before taking action.")
def think(thought: Thought) -> str:
    """Think about something (internal reasoning)."""
    return Think.invoke(airline_data, thought)


@mcp.tool(description="Transfer the conversation to human agents when the request cannot be handled automatically.")
def transfer_to_human_agents(summary: Summary) -> str:
    """Transfer to human agents."""
    return TransferToHumanAgents.invoke(airline_data, summary)


@mcp.tool(description="Update the baggage information of an existing reservation.")
def update_reservation_baggages(
    reservation_id: ReservationId,
    total_baggages: TotalBaggages,
    nonfree_baggages: NonfreeBaggages,
) -> str:
    """Update the baggage information of a reservation."""
    return UpdateReservationBaggages.invoke(
        airline_data, reservation_id, total_baggages, nonfree_baggages
    )


@mcp.tool(description="Update the flights of an existing reservation.")
def update_reservation_flights(
    reservation_id: ReservationId,
    flights: FlightsList,
) -> str:
    """Update the flights of a reservation."""
    flights_dicts = [f.model_dump() for f in flights]
    return UpdateReservationFlights.invoke(airline_data, reservation_id, flights_dicts)


@mcp.tool(description="Update the passengers of an existing reservation.")
def update_reservation_passengers(
    reservation_id: ReservationId,
    passengers: PassengersList,
) -> str:
    """Update the passengers of a reservation."""
    passengers_dicts = [p.model_dump() for p in passengers]
    return UpdateReservationPassengers.invoke(airline_data, reservation_id, passengers_dicts)


async def run_reload_server():
    """Run the reload endpoint on a separate port."""
    reload_app = Starlette(
        routes=[
            Route("/reload", reload_endpoint, methods=["POST", "GET"]),
        ]
    )
    config = uvicorn.Config(reload_app, host="0.0.0.0", port=8001, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    logger.info("Starting Airline Tools MCP Server with SSE (FIXED VERSION)")
    logger.info("MCP endpoint: http://localhost:8000/sse")
    logger.info("Reload endpoint: POST http://localhost:8001/reload")

    # Start the reload server in a separate thread
    def start_reload_server():
        asyncio.run(run_reload_server())

    reload_thread = threading.Thread(target=start_reload_server, daemon=True)
    reload_thread.start()

    # Run the MCP server on port 8000
    mcp.run(transport="sse", port=8000)
