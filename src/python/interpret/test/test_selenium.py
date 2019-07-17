import pytest
from .utils import synthetic_classification
from ..data import ClassHistogram
from ..perf import ROC
from ..glassbox import LogisticRegression, ExplainableBoostingClassifier
from ..visual.interactive import set_show_addr, shutdown_show_server, show_link


@pytest.fixture(scope="module")
def explanations():
    data = synthetic_classification()
    ebm = ExplainableBoostingClassifier()
    ebm.fit(data["train"]["X"], data["train"]["y"])
    lr = LogisticRegression()
    lr.fit(data["train"]["X"], data["train"]["y"])

    hist_exp = ClassHistogram().explain_data(data["train"]["X"], data["train"]["y"])

    lr_global_exp = lr.explain_global()
    lr_local_exp = lr.explain_local(data["test"]["X"].head(), data["test"]["y"].head())
    lr_perf = ROC(lr.predict_proba).explain_perf(data["test"]["X"], data["test"]["y"])

    ebm_global_exp = ebm.explain_global()
    ebm_local_exp = ebm.explain_local(data["test"]["X"], data["test"]["y"])
    ebm_perf = ROC(ebm.predict_proba).explain_perf(data["test"]["X"], data["test"]["y"])

    return [hist_exp, lr_local_exp, lr_global_exp, lr_perf, ebm_local_exp, ebm_global_exp, ebm_perf]


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
