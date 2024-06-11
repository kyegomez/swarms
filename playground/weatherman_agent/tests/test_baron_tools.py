from unittest.mock import patch
from weather_swarm.tools.tools import (
    request_metar_nearest,
    point_query,
    request_ndfd_basic,
)


class TestWeatherFunctions:
    @patch("your_module.request_metar_nearest")
    def test_request_metar_nearest(self, mock_request_metar_nearest):
        mock_request_metar_nearest.return_value = "expected_value"
        result = request_metar_nearest("38", "-96")
        assert result == "expected_value"

    @patch("your_module.point_query")
    def test_point_query_precip_totalaccum(self, mock_point_query):
        mock_point_query.return_value = "expected_value"
        result = point_query(
            "precip-totalaccum-24hr", "Standard-Mercator", -86.6, 34.4
        )
        assert result == "expected_value"

    @patch("your_module.point_query")
    def test_point_query_baron_hires_maxreflectivity(
        self, mock_point_query
    ):
        mock_point_query.return_value = "expected_value"
        result = point_query(
            "baron-hires-maxreflectivity-dbz-all",
            "Mask1-Mercator",
            -86.6,
            34.4,
        )
        assert result == "expected_value"

    @patch("your_module.point_query")
    def test_point_query_baron_hires_windspeed(
        self, mock_point_query
    ):
        mock_point_query.return_value = "expected_value"
        result = point_query(
            "baron-hires-windspeed-mph-10meter",
            "Standard-Mercator",
            -86.6,
            34.4,
        )
        assert result == "expected_value"

    @patch("your_module.request_ndfd_basic")
    def test_request_ndfd_basic(self, mock_request_ndfd_basic):
        mock_request_ndfd_basic.return_value = "expected_value"
        result = request_ndfd_basic(
            34.730301, -86.586098, "forecast_time"
        )
        assert result == "expected_value"
