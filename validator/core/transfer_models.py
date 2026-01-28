"""
Pydantic models for transfer-related data structures
"""

from datetime import datetime
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class TransferData(BaseModel):
    """Individual transfer data model from TaoStats API"""

    id: str = Field(..., description="Transfer ID")
    to_ss58: str = Field(..., description="Recipient SS58 address")
    to_hex: str = Field(..., description="Recipient hex address")
    from_ss58: str = Field(..., description="Sender SS58 address")
    from_hex: str = Field(..., description="Sender hex address")
    network: str = Field(..., description="Network name")
    block_number: int = Field(..., description="Block number")
    timestamp: datetime = Field(..., description="Transfer timestamp")
    amount_rao: int = Field(..., description="Transfer amount in RAO")
    fee_rao: int = Field(..., description="Transaction fee in RAO")
    transaction_hash: str = Field(..., description="Transaction hash")
    extrinsic_id: str = Field(..., description="Extrinsic ID")
    created_at: Optional[datetime] = Field(None, description="When we first saw this transfer")
    updated_at: Optional[datetime] = Field(None, description="When we last updated this record")


class ColdkeyBalance(BaseModel):
    """Coldkey balance tracking model for RAO balances and transfers sent to our target address"""

    coldkey: str = Field(..., description="Coldkey SS58 address")
    balance_rao: int = Field(..., description="Current RAO balance")
    total_sent_rao: int = Field(..., description="Total RAO sent to our target address")
    transfer_count: int = Field(..., description="Number of transfers sent to our target address")
    last_transfer_at: Optional[datetime] = Field(None, description="Timestamp of last transfer to our target address")
    created_at: Optional[datetime] = Field(None, description="When we first saw this coldkey")
    updated_at: Optional[datetime] = Field(None, description="When we last updated this record")


class TransferProcessingState(BaseModel):
    """Transfer processing state model"""

    id: int = Field(1, description="Single row table ID")
    last_processed_timestamp: datetime = Field(..., description="Last timestamp we processed")
    last_processed_block: int = Field(..., description="Last block number we processed")
    processing_interval_hours: int = Field(..., description="How often to check for new transfers")
    target_address: str = Field(..., description="The address we're monitoring for transfers")
    network: str = Field(..., description="Network we're monitoring")
    created_at: Optional[datetime] = Field(None, description="When this record was created")
    updated_at: Optional[datetime] = Field(None, description="When this record was last updated")


class TransferConfig(BaseModel):
    """Configuration for transfer monitoring"""

    api_key: str = Field(..., description="TaoStats API key")
    target_address: str = Field(..., description="Address to monitor for transfers")
    network: str = Field("finney", description="Network to monitor")
    processing_interval_hours: int = Field(24, description="How often to check for new transfers")
    default_lookback_days: int = Field(1, description="Default lookback period in days")
    max_transfers_per_request: int = Field(200, description="Maximum transfers to fetch per API request")


# TaoStats API Models
class AddressInfo(BaseModel):
    """Address information model from TaoStats API"""

    ss58: str = Field(..., description="SS58 formatted address")
    hex: str = Field(..., description="Hex formatted address")


class TaoStatsTransferData(BaseModel):
    """Individual transfer data model from TaoStats API"""

    id: str = Field(..., description="Transfer ID")
    to: AddressInfo = Field(..., description="Recipient address information")
    from_: AddressInfo = Field(..., alias="from", description="Sender address information")
    network: str = Field(..., description="Network name")
    block_number: int = Field(..., description="Block number")
    timestamp: str = Field(..., description="ISO timestamp")
    amount: str = Field(..., description="Transfer amount")
    fee: str = Field(..., description="Transaction fee")
    transaction_hash: str = Field(..., description="Transaction hash")
    extrinsic_id: str = Field(..., description="Extrinsic ID")


class PaginationInfo(BaseModel):
    """Pagination information model from TaoStats API"""

    current_page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")
    next_page: Optional[int] = Field(None, description="Next page number")
    prev_page: Optional[int] = Field(None, description="Previous page number")


class TaoStatsTransferResponse(BaseModel):
    """Complete API response model from TaoStats API"""

    pagination: PaginationInfo = Field(..., description="Pagination information")
    data: List[TaoStatsTransferData] = Field(..., description="List of transfer data")


# API Response Models
class ColdkeyBalanceResponse(BaseModel):
    """API response model for coldkey balance information"""

    coldkey: str = Field(..., description="Coldkey SS58 address")
    balance_rao: int = Field(..., description="Current RAO balance")
    total_sent_rao: int = Field(..., description="Total RAO sent to target address")
    transfer_count: int = Field(..., description="Number of transfers sent to target address")
    last_transfer_at: Optional[datetime] = Field(None, description="Timestamp of last transfer to target address")
    created_at: Optional[datetime] = Field(None, description="When this coldkey was first seen")
    updated_at: Optional[datetime] = Field(None, description="When this record was last updated")


class TournamentFeesResponse(BaseModel):
    """API response model for tournament participation fees"""

    text_tournament_fee_rao: int = Field(..., description="Text tournament participation fee in RAO")
    image_tournament_fee_rao: int = Field(..., description="Image tournament participation fee in RAO")
    environment_tournament_fee_rao: int = Field(..., description="Environment tournament participation fee in RAO")


class BalanceEvent(BaseModel):
    """Balance event model for tracking balance changes tied to tournaments"""

    id: Optional[int] = Field(None, description="Event ID")
    tournament_id: str = Field(..., description="Tournament ID this event is associated with")
    coldkey: str = Field(..., description="Coldkey address affected by this event")
    event_type: str = Field(..., description="Type of balance event: participation_fee, refund, transfer_in")
    amount_rao: int = Field(..., description="Amount in RAO (positive for credits, negative for debits)")
    description: Optional[str] = Field(None, description="Human-readable description of the event")
    created_at: Optional[datetime] = Field(None, description="When this event was created")
    updated_at: Optional[datetime] = Field(None, description="When this event was last updated")


class BalanceEventResponse(BaseModel):
    """API response model for balance events"""

    id: int = Field(..., description="Event ID")
    tournament_id: str = Field(..., description="Tournament ID this event is associated with")
    coldkey: str = Field(..., description="Coldkey address affected by this event")
    event_type: str = Field(..., description="Type of balance event: participation_fee, refund, transfer_in")
    amount_rao: int = Field(..., description="Amount in RAO (positive for credits, negative for debits)")
    description: Optional[str] = Field(None, description="Human-readable description of the event")
    created_at: datetime = Field(..., description="When this event was created")
    updated_at: datetime = Field(..., description="When this event was last updated")
