import pytest
from .utils import synthetic_classification
from ..glassbox import LogisticRegression
from ..visual.interactive import set_show_addr, shutdown_show_server, show_link


@pytest.fixture(scope="module")
def explanations():
    data = synthetic_classification()
    clf = LogisticRegression()
    clf.fit(data["train"]["X"], data["train"]["y"])

    global_exp = clf.explain_global()
    local_exp = clf.explain_local(data["test"]["X"].head(), data["test"]["y"].head())
    return [local_exp, global_exp]


@pytest.mark.selenium
def test_dashboard(explanations):
    from selenium import webdriver
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By

    target_addr = ("127.0.0.1", 7100)
    set_show_addr(target_addr)
    url = show_link(explanations)

    # Set up driver
    driver = webdriver.Firefox()

    # Home page
    driver.get(url)
    driver.implicitly_wait(30)
    driver.find_element_by_id("overview-tab")

    wait = WebDriverWait(driver, 10)
    wait.until(
        EC.text_to_be_present_in_element(
            (By.ID, "overview-tab"), "Welcome to Interpret ML"
        )
    )

    # Move to local
    # Move to global

    # Close driver
    driver.close()

    shutdown_show_server()
