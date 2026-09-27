TEAM_ROUTES = {
    "stock": "stock_team",
    "bad_quality": "quality_team",
    "finance": "finance_team",
}

def route_to_team(category: str) -> str:
    return TEAM_ROUTES.get(category, "unassigned")

