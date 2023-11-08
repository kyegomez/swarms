import os
from typing import Optional
import json
import os
import shutil
import time
import xml.etree.ElementTree as ET
import zipfile
from tempfile import mkdtemp
from typing import Dict, Optional
from urllib.parse import urlparse

import pyautogui
import requests
import semver
import undetected_chromedriver as uc  # type: ignore
import yaml
from extension import load_extension
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from tqdm import tqdm


def _is_blank_agent(agent_name: str) -> bool:
    with open(f"agents/{agent_name}.py", "r") as agent_file:
        agent_data = agent_file.read()
    with open("src/template.py", "r") as template_file:
        template_data = template_file.read()
    return agent_data == template_data


def record(agent_name: str, autotab_ext_path: Optional[str] = None):
    if not os.path.exists("agents"):
        os.makedirs("agents")

    if os.path.exists(
            f"agents/{agent_name}.py") and config.environment != "local":
        if not _is_blank_agent(agent_name=agent_name):
            raise Exception(f"Agent with name {agent_name} already exists")
    driver = get_driver(  # noqa: F841
        autotab_ext_path=autotab_ext_path,
        record_mode=True,
    )
    # Need to keep a reference to the driver so that it doesn't get garbage collected
    with open("src/template.py", "r") as file:
        data = file.read()

    with open(f"agents/{agent_name}.py", "w") as file:
        file.write(data)

    print(
        "\033[34mYou have the Python debugger open, you can run commands in it like you"
        " would in a normal Python shell.\033[0m")
    print(
        "\033[34mTo exit, type 'q' and press enter. For a list of commands type '?' and"
        " press enter.\033[0m")
    breakpoint()


if __name__ == "__main__":
    record("agent")


def extract_domain_from_url(url: str):
    # url = http://username:password@hostname:port/path?arg=value#anchor
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname
    if hostname is None:
        raise ValueError(f"Could not extract hostname from url {url}")
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return hostname


class AutotabChromeDriver(uc.Chrome):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_element_with_retry(self,
                                by=By.ID,
                                value: Optional[str] = None) -> WebElement:
        try:
            return super().find_element(by, value)
        except Exception as e:
            # TODO: Use an LLM to retry, finding a similar element on the DOM
            breakpoint()
            raise e


def open_plugin(driver: AutotabChromeDriver):
    print("Opening plugin sidepanel")
    driver.execute_script("document.activeElement.blur();")
    pyautogui.press("esc")
    pyautogui.hotkey("command", "shift", "y", interval=0.05)  # mypy: ignore


def open_plugin_and_login(driver: AutotabChromeDriver):
    if config.autotab_api_key is not None:
        backend_url = ("http://localhost:8000" if config.environment == "local"
                       else "https://api.autotab.com")
        driver.get(f"{backend_url}/auth/signin-api-key-page")
        response = requests.post(
            f"{backend_url}/auth/signin-api-key",
            json={"api_key": config.autotab_api_key},
        )
        cookie = response.json()
        if response.status_code != 200:
            if response.status_code == 401:
                raise Exception("Invalid API key")
            else:
                raise Exception(
                    f"Error {response.status_code} from backend while logging you in"
                    f" with your API key: {response.text}")
        cookie["name"] = cookie["key"]
        del cookie["key"]
        driver.add_cookie(cookie)

        driver.get("https://www.google.com")
        open_plugin(driver)
    else:
        print("No autotab API key found, heading to autotab.com to sign up")

        url = ("http://localhost:3000/dashboard" if config.environment
               == "local" else "https://autotab.com/dashboard")
        driver.get(url)
        time.sleep(0.5)

        open_plugin(driver)


def get_driver(autotab_ext_path: Optional[str] = None,
               record_mode: bool = False) -> AutotabChromeDriver:
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")  # Necessary for running
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        " (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36")
    options.add_argument("--enable-webgl")
    options.add_argument("--enable-3d-apis")
    options.add_argument("--enable-clipboard-read-write")
    options.add_argument("--disable-popup-blocking")

    if autotab_ext_path is None:
        load_extension()
        options.add_argument("--load-extension=./src/extension/autotab")
    else:
        options.add_argument(f"--load-extension={autotab_ext_path}")

    options.add_argument("--allow-running-insecure-content")
    options.add_argument("--disable-web-security")
    options.add_argument(f"--user-data-dir={mkdtemp()}")
    options.binary_location = config.chrome_binary_location
    driver = AutotabChromeDriver(options=options)
    if record_mode:
        open_plugin_and_login(driver)

    return driver


class SiteCredentials(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None
    login_with_google_account: Optional[str] = None
    login_url: Optional[str] = None

    def __init__(self, **data) -> None:
        super().__init__(**data)
        if self.name is None:
            self.name = self.email


class GoogleCredentials(BaseModel):
    credentials: Dict[str, SiteCredentials]

    def __init__(self, **data) -> None:
        super().__init__(**data)
        for cred in self.credentials.values():
            cred.login_url = "https://accounts.google.com/v3/signin"

    @property
    def default(self) -> SiteCredentials:
        if "default" not in self.credentials:
            if len(self.credentials) == 1:
                return list(self.credentials.values())[0]
            raise Exception("No default credentials found in config")
        return self.credentials["default"]


class Config(BaseModel):
    autotab_api_key: Optional[str]
    credentials: Dict[str, SiteCredentials]
    google_credentials: GoogleCredentials
    chrome_binary_location: str
    environment: str

    @classmethod
    def load_from_yaml(cls, path: str):
        with open(path, "r") as config_file:
            config = yaml.safe_load(config_file)
            _credentials = {}
            for domain, creds in config.get("credentials", {}).items():
                if "login_url" not in creds:
                    creds["login_url"] = f"https://{domain}/login"
                site_creds = SiteCredentials(**creds)
                _credentials[domain] = site_creds
                for alt in creds.get("alts", []):
                    _credentials[alt] = site_creds

            google_credentials = {}
            for creds in config.get("google_credentials", []):
                credentials: SiteCredentials = SiteCredentials(**creds)
                google_credentials[credentials.name] = credentials

            chrome_binary_location = config.get("chrome_binary_location")
            if chrome_binary_location is None:
                raise Exception("Must specify chrome_binary_location in config")

            autotab_api_key = config.get("autotab_api_key")
            if autotab_api_key == "...":
                autotab_api_key = None

            return cls(
                autotab_api_key=autotab_api_key,
                credentials=_credentials,
                google_credentials=GoogleCredentials(
                    credentials=google_credentials),
                chrome_binary_location=config.get("chrome_binary_location"),
                environment=config.get("environment", "prod"),
            )

    def get_site_credentials(self, domain: str) -> SiteCredentials:
        credentials = self.credentials[domain].copy()
        return credentials


config = Config.load_from_yaml(".autotab.yaml")


def is_signed_in_to_google(driver):
    cookies = driver.get_cookies()
    return len([c for c in cookies if c["name"] == "SAPISID"]) != 0


def google_login(driver,
                 credentials: Optional[SiteCredentials] = None,
                 navigate: bool = True):
    print("Logging in to Google")
    if navigate:
        driver.get("https://accounts.google.com/")
        time.sleep(1)
        if is_signed_in_to_google(driver):
            print("Already signed in to Google")
            return

    if os.path.exists("google_cookies.json"):
        print("cookies exist, doing loading")
        with open("google_cookies.json", "r") as f:
            google_cookies = json.load(f)
            for cookie in google_cookies:
                if "expiry" in cookie:
                    cookie["expires"] = cookie["expiry"]
                    del cookie["expiry"]
                driver.execute_cdp_cmd("Network.setCookie", cookie)
            time.sleep(1)
            driver.refresh()
            time.sleep(2)

    if not credentials:
        credentials = config.google_credentials.default

    if credentials is None:
        raise Exception("No credentials provided for Google login")

    email_input = driver.find_element(By.CSS_SELECTOR, "[type='email']")
    email_input.send_keys(credentials.email)
    email_input.send_keys(Keys.ENTER)
    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "[type='password']")))

    password_input = driver.find_element(By.CSS_SELECTOR, "[type='password']")
    password_input.send_keys(credentials.password)
    password_input.send_keys(Keys.ENTER)
    time.sleep(1.5)
    print("Successfully logged in to Google")

    cookies = driver.get_cookies()
    if not is_signed_in_to_google(driver):
        # Probably wanted to have us solve a captcha, or 2FA or confirm recovery details
        print("Need 2FA help to log in to Google")
        # TODO: Show screenshot it to the user
        breakpoint()

    if not os.path.exists("google_cookies.json"):
        print("Setting Google cookies for future use")
        # Log out to have access to the right cookies
        driver.get("https://accounts.google.com/Logout")
        time.sleep(2)
        cookies = driver.get_cookies()
        cookie_names = ["__Host-GAPS", "SMSV", "NID", "ACCOUNT_CHOOSER"]
        google_cookies = [
            cookie for cookie in cookies
            if cookie["domain"] in [".google.com", "accounts.google.com"] and
            cookie["name"] in cookie_names
        ]
        with open("google_cookies.json", "w") as f:
            json.dump(google_cookies, f)

        # Log back in
        login_button = driver.find_element(
            By.CSS_SELECTOR, f"[data-identifier='{credentials.email}']")
        login_button.click()
        time.sleep(1)
        password_input = driver.find_element(By.CSS_SELECTOR,
                                             "[type='password']")
        password_input.send_keys(credentials.password)
        password_input.send_keys(Keys.ENTER)

        time.sleep(3)
        print("Successfully copied Google cookies for the future")


def login(driver, url: str):
    domain = extract_domain_from_url(url)

    credentials = config.get_site_credentials(domain)
    login_url = credentials.login_url
    if credentials.login_with_google_account:
        google_credentials = config.google_credentials.credentials[
            credentials.login_with_google_account]
        _login_with_google(driver, login_url, google_credentials)
    else:
        _login(driver, login_url, credentials=credentials)


def _login(driver, url: str, credentials: SiteCredentials):
    print(f"Logging in to {url}")
    driver.get(url)
    time.sleep(2)
    email_input = driver.find_element(By.NAME, "email")
    email_input.send_keys(credentials.email)
    password_input = driver.find_element(By.NAME, "password")
    password_input.send_keys(credentials.password)
    password_input.send_keys(Keys.ENTER)

    time.sleep(3)
    print(f"Successfully logged in to {url}")


def _login_with_google(driver, url: str, google_credentials: SiteCredentials):
    print(f"Logging in to {url} with Google")

    google_login(driver, credentials=google_credentials)

    driver.get(url)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body")))

    main_window = driver.current_window_handle
    xpath = (
        "//*[contains(text(), 'Continue with Google') or contains(text(), 'Sign in with"
        " Google') or contains(@title, 'Sign in with Google')]")

    WebDriverWait(driver,
                  10).until(EC.presence_of_element_located((By.XPATH, xpath)))
    driver.find_element(
        By.XPATH,
        xpath,
    ).click()

    driver.switch_to.window(driver.window_handles[-1])
    driver.find_element(
        By.XPATH,
        f"//*[contains(text(), '{google_credentials.email}')]").click()

    driver.switch_to.window(main_window)

    time.sleep(5)
    print(f"Successfully logged in to {url}")


def update():
    print("updating extension...")
    # Download the autotab.crx file
    response = requests.get(
        "https://github.com/Planetary-Computers/autotab-extension/raw/main/autotab.crx",
        stream=True,
    )

    # Check if the directory exists, if not create it
    if os.path.exists("src/extension/.autotab"):
        shutil.rmtree("src/extension/.autotab")
    os.makedirs("src/extension/.autotab")

    # Open the file in write binary mode
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open("src/extension/.autotab/autotab.crx", "wb") as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        print("ERROR, something went wrong")

    # Unzip the file
    with zipfile.ZipFile("src/extension/.autotab/autotab.crx", "r") as zip_ref:
        zip_ref.extractall("src/extension/.autotab")
    os.remove("src/extension/.autotab/autotab.crx")
    if os.path.exists("src/extension/autotab"):
        shutil.rmtree("src/extension/autotab")
    os.rename("src/extension/.autotab", "src/extension/autotab")


def should_update():
    if not os.path.exists("src/extension/autotab"):
        return True
    # Fetch the XML file
    response = requests.get(
        "https://raw.githubusercontent.com/Planetary-Computers/autotab-extension/main/update.xml"
    )
    xml_content = response.content

    # Parse the XML file
    root = ET.fromstring(xml_content)
    namespaces = {
        "ns": "http://www.google.com/update2/response"
    }  # add namespaces
    xml_version = root.find(".//ns:app/ns:updatecheck",
                            namespaces).get("version")

    # Load the local JSON file
    with open("src/extension/autotab/manifest.json", "r") as f:
        json_content = json.load(f)
    json_version = json_content["version"]
    # Compare versions
    return semver.compare(xml_version, json_version) > 0


def load_extension():
    should_update() and update()


if __name__ == "__main__":
    print("should update:", should_update())
    update()


def play(agent_name: Optional[str] = None):
    if agent_name is None:
        agent_files = os.listdir("agents")
        if len(agent_files) == 0:
            raise Exception("No agents found in agents/ directory")
        elif len(agent_files) == 1:
            agent_file = agent_files[0]
        else:
            print("Found multiple agent files, please select one:")
            for i, file in enumerate(agent_files, start=1):
                print(f"{i}. {file}")

            selected = int(input("Select a file by number: ")) - 1
            agent_file = agent_files[selected]
    else:
        agent_file = f"{agent_name}.py"

    os.system(f"python agents/{agent_file}")


if __name__ == "__main__":
    play()
"""


chrome_binary_location: /Applications/Google Chrome.app/Contents/MacOS/Google Chrome

autotab_api_key: ... # Go to https://autotab.com/dashboard to get your API key, or
# run `autotab record` with this field blank and you will be prompted to log in to autotab

# Optional, programmatically login to services using "Login with Google" authentication
google_credentials:
  - name: default
    email: ...
    password: ...

  # Optional, specify alternative accounts to use with Google login on a per-service basis
  - email: you@gmail.com # Credentials without a name use email as key
    password: ...

credentials:
  notion.so:
    alts:
    - notion.com
    login_with_google_account: default

  figma.com:
    email: ...
    password: ...

  airtable.com:
    login_with_google_account: you@gmail.com
"""
