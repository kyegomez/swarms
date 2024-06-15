def calculate_monthly_charge(
    development_time_hours: float,
    hourly_rate: float,
    amortization_months: int,
    api_calls_per_month: int,
    cost_per_api_call: float,
    monthly_maintenance: float,
    additional_monthly_costs: float,
    profit_margin_percentage: float,
) -> float:
    """
    Calculate the monthly charge for a service based on various cost factors.

    Parameters:
    - development_time_hours (float): The total number of hours spent on development and setup.
    - hourly_rate (float): The rate per hour for development and setup.
    - amortization_months (int): The number of months over which to amortize the development and setup costs.
    - api_calls_per_month (int): The number of API calls made per month.
    - cost_per_api_call (float): The cost per API call.
    - monthly_maintenance (float): The monthly maintenance cost.
    - additional_monthly_costs (float): Any additional monthly costs.
    - profit_margin_percentage (float): The desired profit margin as a percentage.

    Returns:
    - monthly_charge (float): The calculated monthly charge for the service.
    """

    # Calculate Development and Setup Costs (amortized monthly)
    development_and_setup_costs_monthly = (
        development_time_hours * hourly_rate
    ) / amortization_months

    # Calculate Operational Costs per Month
    operational_costs_monthly = (
        (api_calls_per_month * cost_per_api_call)
        + monthly_maintenance
        + additional_monthly_costs
    )

    # Calculate Total Monthly Costs
    total_monthly_costs = (
        development_and_setup_costs_monthly
        + operational_costs_monthly
    )

    # Calculate Pricing with Profit Margin
    monthly_charge = total_monthly_costs * (
        1 + profit_margin_percentage / 100
    )

    return monthly_charge


# Example usage:
monthly_charge = calculate_monthly_charge(
    development_time_hours=100,
    hourly_rate=500,
    amortization_months=12,
    api_calls_per_month=500000,
    cost_per_api_call=0.002,
    monthly_maintenance=1000,
    additional_monthly_costs=300,
    profit_margin_percentage=10000,
)

print(f"Monthly Charge: ${monthly_charge:.2f}")
