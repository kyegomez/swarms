# -*- coding: utf-8 -*-
import requests
from typing import Tuple, Optional, List
import json
from urllib.parse import quote
import urllib
import hashlib
import os
import json
from ...tool import Tool


class BaiduMapAPI:
    def __init__(self, ak: str, sk: str):
        self.ak = ak
        self.sk = sk
        self.base_url = "http://api.map.baidu.com"

    def generate_url_with_sn(self, url: str) -> str:
        """
        生成百度地图API请求中的SN码
        :param url: API请求的URL
        :return: SN码
        """
        query_str = url[len("http://api.map.baidu.com/") - 1 :]
        encoded_str = quote(query_str, safe="/:=&?#+!$,;'@()*[]")
        raw_str = encoded_str + self.sk
        sn = hashlib.md5(urllib.parse.quote_plus(raw_str).encode("utf-8")).hexdigest()
        url_with_sn = f"{url}&sn={sn}"
        return url_with_sn

    def get_location(self, address: str) -> Optional[Tuple[float, float]]:
        """
        该函数仅适用于中国境内地图（内陆），this function only suitable for locations in Mainland China
        根据地址获取地点经纬度, return the coordinates
        :param address: 地址
        :return: 地点经纬度 (纬度, 经度)，若无法获取则返回None; return (latitute, longitute)
        """
        url = f"{self.base_url}/geocoding/v3/?address={address}&output=json&ak={self.ak}&callback=showLocation"
        url = self.generate_url_with_sn(url)
        response = requests.get(url)
        json_text = response.text[len("showLocation&&showLocation(") : -1]
        data = json.loads(json_text)
        if data["status"] == 0:
            result = data["result"]
            location = result["location"]
            return location["lat"], location["lng"]
        else:
            return None

    def get_address_by_coordinates(self, lat: float, lng: float) -> Optional[str]:
        """
        该函数仅适用于中国境内地图（内陆），this function only suitable for locations in Mainland China
        根据经纬度获取地点名称
        :param lat: 纬度
        :param lng: 经度
        :return: 地点名称列表，包含经纬度，具体地址等信息，若无法获取则返回None
        """
        url = f"{self.base_url}/reverse_geocoding/v3/?location={lat},{lng}&output=json&ak={self.ak}"
        url = self.generate_url_with_sn(url)
        response = requests.get(url)
        data = response.json()
        if data["status"] == 0:
            result = data["result"]["formatted_address"]
            return result
        else:
            return None

    def get_nearby_places(
        self, location: Tuple[float, float], radius: int, keyword: Optional[str] = "餐厅"
    ) -> List[str]:
        """
        该函数仅适用于中国境内地图（内陆），this function only suitable for locations in Mainland China
        查询某位置附近的地点
        :param location: 地点的经纬度 (纬度, 经度)
        :param radius: 查询半径（单位：米）
        :param keyword: 查询关键词（必选）
        :return: 地点名称列表
        """
        lat, lng = location
        url = f"{self.base_url}/place/v2/search?query={keyword}&location={lat},{lng}&radius={radius}&output=json&ak={self.ak}"
        if keyword:
            url += f"&query={keyword}"
        url = self.generate_url_with_sn(url)

        response = requests.get(url)
        data = response.json()
        if data["status"] == 0:
            results = data["results"]
            place_names = [result["name"] for result in results]
            return place_names
        else:
            return []

    def get_route(self, origin: str, destination: str) -> Optional[List[str]]:
        """
        该函数仅适用于中国境内地图（内陆），this function only suitable for locations in Mainland China
        查询两地之间的路线
        :param origin: 起点地址
        :param destination: 终点地址
        :return: 路线描述列表，若无法获取则返回None
        """
        origin_location = self.get_location(origin)
        destination_location = self.get_location(destination)
        if origin_location and destination_location:
            origin_lat, origin_lng = origin_location
            destination_lat, destination_lng = destination_location
            url = f"{self.base_url}/direction/v2/transit?origin={origin_lat},{origin_lng}&destination={destination_lat},{destination_lng}&output=json&ak={self.ak}"
            url = self.generate_url_with_sn(url)
            response = requests.get(url)
            data = response.json()
            if data["status"] == 0:
                routes = data["result"]["routes"]
                instructions = []
                for route in routes:
                    for step in route["steps"]:
                        for item in step:
                            instructions.append(item["instructions"])
                return instructions
        return None

    def get_distance(self, origin: str, destination: str) -> float:
        """
        该函数仅适用于中国境内地图（内陆），this function only suitable for locations in Mainland China
        查询两地之间的距离
        :param origin: 起点地址
        :param destination: 终点地址
        :return: 两地之间距离(米)
        """
        origin_location = self.get_location(origin)
        destination_location = self.get_location(destination)
        if origin_location and destination_location:
            origin_lat, origin_lng = origin_location
            destination_lat, destination_lng = destination_location
            url = f"{self.base_url}/direction/v2/transit?origin={origin_lat},{origin_lng}&destination={destination_lat},{destination_lng}&output=json&ak={self.ak}"
            url = self.generate_url_with_sn(url)
            response = requests.get(url)
            data = response.json()
            if data["status"] == 0:
                distance = data["result"]["routes"][0]["distance"]
                return distance
        return None


def build_tool(config) -> Tool:
    tool = Tool(
        "Map Info",
        "Look up map information in China",
        name_for_model="Baidu_Map",
        description_for_model="Plugin for look up map information in China",
        logo_url="https://your-app-url.com/.well-known/logo.png",
        contact_email="hello@contact.com",
        legal_info_url="hello@legal.com",
    )

    KEY = config["subscription_key"]
    SK = config["baidu_secret_key"]
    baidu_map = BaiduMapAPI(KEY, SK)

    @tool.get("/get_location")
    def get_location(address: str) -> Optional[Tuple[float, float]]:
        """
        该函数仅适用于中国境内地图（内陆），this function only suitable for locations in Mainland China
        根据地址获取地点经纬度, return the coordinates
        :param address: 地址
        :return: 地点经纬度 (纬度, 经度)，若无法获取则返回None; return (latitute, longitute)
        """
        return baidu_map.get_location(address)

    @tool.get("/get_address_by_coordinates")
    def get_address_by_coordinates(lat: float, lng: float) -> Optional[str]:
        """
        该函数仅适用于中国境内地图（内陆），this function only suitable for locations in Mainland China
        根据经纬度获取地点名称
        :param lat: 纬度
        :param lng: 经度
        :return: 地点名称列表，包含经纬度，具体地址等信息，若无法获取则返回None
        """
        return baidu_map.get_address_by_coordinates(lat, lng)

    @tool.get("/get_nearby_places")
    def get_nearby_places(
        location: Tuple[float, float], radius: int, keyword: Optional[str] = "餐厅"
    ) -> List[str]:
        """
        该函数仅适用于中国境内地图（内陆），this function only suitable for locations in Mainland China
        查询某位置附近的地点
        :param location: 地点的经纬度 (纬度, 经度)
        :param radius: 查询半径（单位：米）
        :param keyword: 查询关键词（必选）
        :return: 地点名称列表
        """
        return baidu_map.get_nearby_places(location, radius, keyword)

    @tool.get("/get_route")
    def get_route(self, origin: str, destination: str) -> Optional[List[str]]:
        """
        该函数仅适用于中国境内地图（内陆），this function only suitable for locations in Mainland China
        查询两地之间的路线
        :param origin: 起点地址
        :param destination: 终点地址
        :return: 路线描述列表，若无法获取则返回None
        """
        return baidu_map.get_route(origin, destination)

    @tool.get("/get_distance")
    def get_distance(self, origin: str, destination: str) -> float:
        """
        该函数仅适用于中国境内地图（内陆），this function only suitable for locations in Mainland China
        查询两地之间的距离
        :param origin: 起点地址
        :param destination: 终点地址
        :return: 两地之间距离(米)
        """
        return baidu_map.get_distance(origin, destination)

    return tool
