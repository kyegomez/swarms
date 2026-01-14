from swarms.tcandon_router import TCAndonRouter


def test_selects_weather():
    agents = [
        {"id": "weather", "name": "WeatherAgent", "description": "Provides weather forecasts and alerts."},
        {"id": "maps", "name": "MapsAgent", "description": "Helps with directions and points of interest."},
    ]

    router = TCAndonRouter(run="test", max_agents=2, oos_threshold=0.01)
    res = router.run(agents, "What's the weather like today in London?")

    assert res["selected"], "Expected at least one selected agent"
    assert res["selected"][0] == "weather"


def test_oos_when_threshold_high():
    agents = [
        {"id": "a1", "description": "Handles billing and invoices."},
    ]
    # Make threshold high so no agent qualifies
    router = TCAndonRouter(run="test", max_agents=1, oos_threshold=0.99)
    res = router.run(agents, "Tell me a joke about cats")

    assert res["selected"] == [], "Expected no selected agents (oos)"
