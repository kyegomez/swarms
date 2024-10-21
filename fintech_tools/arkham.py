import requests
from bs4 import BeautifulSoup


def scrape_blackrock_trades():
    url = "https://arkhamintelligence.com/blackrock/trades"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")

        # Example: Assuming trades are in a table
        trades = []
        table = soup.find("table", {"id": "trades-table"})

        if table:
            for row in table.find_all("tr"):
                columns = row.find_all("td")
                if len(columns) > 0:
                    trade = {
                        "trade_date": columns[0].text.strip(),
                        "asset": columns[1].text.strip(),
                        "action": columns[2].text.strip(),
                        "quantity": columns[3].text.strip(),
                        "price": columns[4].text.strip(),
                        "total_value": columns[5].text.strip(),
                    }
                    trades.append(trade)
        return trades
    else:
        print(
            f"Failed to fetch data. Status code: {response.status_code}"
        )
        return None


if __name__ == "__main__":
    trades = scrape_blackrock_trades()
    if trades:
        for trade in trades:
            print(trade)
