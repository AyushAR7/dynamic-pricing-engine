from pydantic import BaseModel, Field
from typing import Dict

class RideRequest(BaseModel):
    Number_of_Riders: int = Field(..., ge=1, description="Number of riders requesting the ride")
    Number_of_Drivers: int = Field(..., ge=1, description="Number of available drivers")
    Location_Category: str = Field(..., example="Urban", description="Urban, Suburban, or Rural")
    Customer_Loyalty_Status: str = Field(..., example="Silver", description="Regular, Silver, or Gold")
    Number_of_Past_Rides: int = Field(..., ge=0, example=25)
    Average_Ratings: float = Field(..., ge=1.0, le=5.0, example=4.5)
    Time_of_Booking: str = Field(..., example="Night", description="Morning, Afternoon, Evening, or Night")
    Vehicle_Type: str = Field(..., example="Premium", description="Economy or Premium")
    Expected_Ride_Duration: float = Field(..., gt=0, example=35.0)

class BasePriceResponse(BaseModel):
    base_price: float

class DynamicPriceResponse(BaseModel):
    base_price: float
    demand_supply_ratio: float
    surge_multiplier: float
    dynamic_price: float

class ExplainResponse(BaseModel):
    base_price: float
    expected_value: float
    shap_contributions: Dict[str, float]