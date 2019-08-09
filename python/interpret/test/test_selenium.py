import pytest
from .utils import synthetic_classification, get_all_explainers
from ..data import ClassHistogram
from ..perf import ROC
from ..glassbox import LogisticRegression, ExplainableBoostingClassifier
from ..visual.interactive import set_show_addr, shutdown_show_server, show_link

# Timeout for element to not show up in selenium driver.
TIMEOUT = 60


@pytest.fixture(scope="module")
def driver():
    from selenium import webdriver

    _driver = webdriver.Firefox()

    # Set up driver
    _driver = webdriver.Firefox()
    _driver.implicitly_wait(TIMEOUT)

    yield _driver

    # Close driver
    _driver.close()


@pytest.fixture(scope="module")
def all_explanations():
    all_explainers = get_all_explainers()
    data = synthetic_classification()
    blackbox = LogisticRegression()
    blackbox.fit(data["train"]["X"], data["train"]["y"])

    explanations = []
    predict_fn = lambda x: blackbox.predict_proba(x)  # noqa: E731
    for explainer_class in all_explainers:
        if explainer_class.explainer_type == "blackbox":
            explainer = explainer_class(predict_fn, data["train"]["X"])
        elif explainer_class.explainer_type == "model":
            explainer = explainer_class()
            explainer.fit(data["train"]["X"], data["train"]["y"])
        elif explainer_class.explainer_type == "data":
            explainer = explainer_class()
        elif explainer_class.explainer_type == "perf":
            explainer = explainer_class(predict_fn)
        else:
            raise Exception("Not supported explainer type.")

        if "local" in explainer.available_explanations:
            # With labels
            explanation = explainer.explain_local(
                data["test"]["X"].head(), data["test"]["y"].head()
            )
            explanations.append(explanation)

            # Without labels
            explanation = explainer.explain_local(data["test"]["X"].head())
            explanations.append(explanation)
        if "global" in explainer.available_explanations:
            explanation = explainer.explain_global()
            explanations.append(explanation)
        if "data" in explainer.available_explanations:
            explanation = explainer.explain_data(data["train"]["X"], data["train"]["y"])
            explanations.append(explanation)
        if "perf" in explainer.available_explanations:
            explanation = explainer.explain_perf(data["test"]["X"], data["test"]["y"])
            explanations.append(explanation)

    return explanations


@pytest.fixture(scope="module")
def small_explanations():
    data = synthetic_classification()
    ebm = ExplainableBoostingClassifier()
    ebm.fit(data["train"]["X"], data["train"]["y"])
    lr = LogisticRegression()
    lr.fit(data["train"]["X"], data["train"]["y"])

    hist_exp = ClassHistogram().explain_data(data["train"]["X"], data["train"]["y"], name="Histogram")

    lr_global_exp = lr.explain_global(name="LR")
    lr_local_exp = lr.explain_local(data["test"]["X"].head(), data["test"]["y"].head(), name="LR")
    lr_perf = ROC(lr.predict_proba).explain_perf(data["test"]["X"], data["test"]["y"], name="LR")

    ebm_global_exp = ebm.explain_global(name="EBM")
    ebm_local_exp = ebm.explain_local(data["test"]["X"].head(), data["test"]["y"].head(), name="EBM")
    ebm_perf = ROC(ebm.predict_proba).explain_perf(data["test"]["X"], data["test"]["y"], name="EBM")

    return [hist_exp, lr_local_exp, lr_global_exp, lr_perf, ebm_local_exp, ebm_global_exp, ebm_perf]


# TODO: Code duplication, refactor.
@pytest.mark.selenium
def test_show_small_set_selenium(small_explanations, driver):
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By

    target_addr = ("127.0.0.1", 7100)
    set_show_addr(target_addr)
    dashboard_url = show_link(small_explanations)
    mini_url = show_link(small_explanations[2])

    # Home page
    driver.get(dashboard_url)
    driver.find_element_by_id("overview-tab")

    # Expect overview tab's welcome message
    wait = WebDriverWait(driver, TIMEOUT)
    wait.until(
        EC.text_to_be_present_in_element(
            (By.ID, "overview-tab"), "Welcome to Interpret ML"
        )
    )

    # Move to data
    tabs_el = driver.find_element_by_id("tabs")
    data_tab_el = tabs_el.find_element_by_xpath("//span[contains(text(),'Data')]")
    data_tab_el.click()

    # Expect dropdown
    wait.until(
        EC.text_to_be_present_in_element(
            (By.CLASS_NAME, "card-title"), "Select Explanation"
        )
    )
    # Click on dropdown
    data_tab_el = driver.find_element_by_id("data-tab")
    dropdown_el = data_tab_el.find_element_by_class_name("Select-placeholder")
    dropdown_el.click()
    # Select histogram item
    hist_el = dropdown_el.find_element_by_xpath("//div[contains(text(),'Histogram')]")
    hist_el.click()

    # Expect shared container
    wait.until(
        EC.presence_of_element_located(
            (By.XPATH, "//div[contains(text(),'Select Components to Graph')]")
        )
    )
    first_path = "//label[@for='checkbox0']"
    second_path = "//label[@for='checkbox1']"
    wait.until(
        EC.presence_of_element_located(
            (By.XPATH, first_path)
        )
    )
    # Click on first two items
    first_el = data_tab_el.find_element_by_xpath(first_path)
    first_el.click()
    second_el = data_tab_el.find_element_by_xpath(second_path)
    second_el.click()
    # Expect two specific graphs
    wait.until(EC.presence_of_element_located(
        (By.ID, "graph-0-0")
    ))
    wait.until(EC.presence_of_element_located(
        (By.ID, "graph-0-1")
    ))
    # Expect overall graph
    wait.until(EC.presence_of_element_located(
        (By.ID, "example-overall-graph-0")
    ))

    # Move to perf
    tabs_el = driver.find_element_by_id("tabs")
    perf_tab_el = tabs_el.find_element_by_xpath("//span[contains(text(),'Perf')]")
    perf_tab_el.click()

    # Expect dropdown
    wait.until(
        EC.text_to_be_present_in_element(
            (By.CLASS_NAME, "card-title"), "Select Explanation"
        )
    )
    # Click on dropdown
    perf_tab_el = driver.find_element_by_id("perf-tab")
    # Select LR item
    dropdown_el = perf_tab_el.find_element_by_class_name("Select-control")
    dropdown_el.click()
    lr_el = dropdown_el.find_element_by_xpath("//div[contains(text(),'LR')]")
    lr_el.click()
    # Select EBM item
    dropdown_el.click()
    ebm_el = dropdown_el.find_element_by_xpath("//div[contains(text(),'EBM')]")
    ebm_el.click()
    # Expect two overall graphs
    wait.until(
        EC.presence_of_element_located(
            (By.XPATH, "//div[@id='perf-overall-plot-container-0']//div[contains(@id, 'example-overall-graph-')]")
        )
    )
    wait.until(
        EC.presence_of_element_located(
            (By.XPATH, "//div[@id='perf-overall-plot-container-1']//div[contains(@id, 'example-overall-graph-')]")
        )
    )

    # Move to global
    tabs_el = driver.find_element_by_id("tabs")
    global_tab_el = tabs_el.find_element_by_xpath("//span[contains(text(),'Global')]")
    global_tab_el.click()

    # Expect dropdown
    wait.until(
        EC.text_to_be_present_in_element(
            (By.CLASS_NAME, "card-title"), "Select Explanation"
        )
    )

    # Click on dropdown
    global_tab_el = driver.find_element_by_id("global-tab")
    # Select LR item
    dropdown_el = global_tab_el.find_element_by_class_name("Select-control")
    dropdown_el.click()
    lr_el = dropdown_el.find_element_by_xpath("//div[contains(text(),'LR')]")
    lr_el.click()
    # Select EBM item
    dropdown_el.click()
    ebm_el = dropdown_el.find_element_by_xpath("//div[contains(text(),'EBM')]")
    ebm_el.click()

    # Expect shared container
    wait.until(
        EC.presence_of_element_located(
            (By.XPATH, "//div[contains(text(),'Select Components to Graph')]")
        )
    )
    first_path = "//label[@for='checkbox0']"
    second_path = "//label[@for='checkbox1']"
    wait.until(
        EC.presence_of_element_located(
            (By.XPATH, first_path)
        )
    )
    # Click on first two items
    first_el = global_tab_el.find_element_by_xpath(first_path)
    first_el.click()
    second_el = global_tab_el.find_element_by_xpath(second_path)
    second_el.click()
    # Expect four specific graphs
    wait.until(EC.presence_of_element_located(
        (By.ID, "graph-2-0")
    ))
    wait.until(EC.presence_of_element_located(
        (By.ID, "graph-2-1")
    ))
    wait.until(EC.presence_of_element_located(
        (By.ID, "graph-5-0")
    ))
    wait.until(EC.presence_of_element_located(
        (By.ID, "graph-5-1")
    ))
    # Expect two overall graphs
    wait.until(EC.presence_of_element_located(
        (By.ID, "example-overall-graph-2")
    ))
    wait.until(EC.presence_of_element_located(
        (By.ID, "example-overall-graph-5")
    ))

    # Move to local
    tabs_el = driver.find_element_by_id("tabs")
    local_tab_el = tabs_el.find_element_by_xpath("//span[contains(text(),'Local')]")
    local_tab_el.click()

    # Expect dropdown
    wait.until(
        EC.text_to_be_present_in_element(
            (By.CLASS_NAME, "card-title"), "Select Explanation"
        )
    )

    # Click on dropdown
    local_tab_el = driver.find_element_by_id("local-tab")
    # Select LR item
    dropdown_el = local_tab_el.find_element_by_class_name("Select-control")
    dropdown_el.click()
    lr_el = dropdown_el.find_element_by_xpath("//div[contains(text(),'LR')]")
    lr_el.click()
    # Select EBM item
    dropdown_el.click()
    ebm_el = dropdown_el.find_element_by_xpath("//div[contains(text(),'EBM')]")
    ebm_el.click()

    # Expect two selectors
    wait.until(
        EC.presence_of_element_located(
            (By.XPATH, "//div[@class='gr']/div[@class='gr-col'][1]")
        )
    )
    wait.until(
        EC.presence_of_element_located(
            (By.XPATH, "//div[@class='gr']/div[@class='gr-col'][2]")
        )
    )
    # Click on first two items for both panes
    for pane_idx in range(2):
        for item_idx in range(2):
            item_path = "//div[@class='gr-col'][{0}]//label[@for='checkbox{1}']".format(pane_idx + 1, item_idx)
            wait.until(
                EC.presence_of_element_located(
                    (By.XPATH, item_path)
                )
            )
            item_el = local_tab_el.find_element_by_xpath(item_path)
            item_el.click()
    # Expect four specific graphs
    wait.until(EC.presence_of_element_located(
        (By.ID, "graph-1-0")
    ))
    wait.until(EC.presence_of_element_located(
        (By.ID, "graph-1-1")
    ))
    wait.until(EC.presence_of_element_located(
        (By.ID, "graph-4-0")
    ))
    wait.until(EC.presence_of_element_located(
        (By.ID, "graph-4-1")
    ))
    # Expect two overall graphs
    wait.until(EC.presence_of_element_located(
        (By.ID, "example-overall-graph-1")
    ))
    wait.until(EC.presence_of_element_located(
        (By.ID, "example-overall-graph-4")
    ))

    # Mini page
    driver.get(mini_url)
    driver.find_element_by_class_name("card")
    # Expect overall graph
    wait.until(EC.presence_of_element_located(
        (By.ID, "example-overall-graph--1")
    ))
    # Click on specific graph
    dropdown_el = driver.find_element_by_class_name("Select-control")
    dropdown_el.click()
    specific_el = dropdown_el.find_element_by_xpath("//div[contains(text(),'1 : ')]")
    specific_el.click()
    # Expect specific graph
    wait.until(EC.presence_of_element_located(
        (By.ID, "graph-0-0")
    ))

    shutdown_show_server()
