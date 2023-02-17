import pytest
from sklearn.ensemble import RandomForestClassifier

from interpret.test.utils import synthetic_classification, get_all_explainers
from interpret.glassbox import LogisticRegression
from interpret.glassbox.decisiontree import TreeExplanation

# from interpret.blackbox import PermutationImportance
from interpret.visual.interactive import set_show_addr, shutdown_show_server, show_link
from copy import deepcopy
import os

# Timeout for element to not show up in selenium driver.
TIMEOUT = 60


@pytest.fixture(scope="module")
def all_explanations():
    all_explainers = get_all_explainers()
    data = synthetic_classification()
    blackbox = LogisticRegression()
    blackbox.fit(data["train"]["X"], data["train"]["y"])
    tree = RandomForestClassifier()
    tree.fit(data["train"]["X"], data["train"]["y"])

    explanations = []
    predict_fn = lambda x: blackbox.predict_proba(x)  # noqa: E731
    for explainer_class in all_explainers:
        # if explainer_class == PermutationImportance:
        #     explainer = explainer_class(predict_fn, data["train"]["X"], data["train"]["y"])
        if explainer_class.explainer_type == "blackbox":
            explainer = explainer_class(predict_fn, data["train"]["X"])
        elif explainer_class.explainer_type == "model":
            explainer = explainer_class()
            explainer.fit(data["train"]["X"], data["train"]["y"])
        elif explainer_class.explainer_type == "specific":
            explainer = explainer_class(tree, data["train"]["X"])
        elif explainer_class.explainer_type == "data":
            explainer = explainer_class()
        elif explainer_class.explainer_type == "perf":
            explainer = explainer_class(predict_fn)
        else:
            raise Exception("Not supported explainer type.")

        if "local" in explainer.available_explanations:
            explanation = explainer.explain_local(
                data["test"]["X"].head(), data["test"]["y"].head()
            )
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


num_jobs = int(os.getenv("PYTEST_XDIST_WORKER_COUNT", 1))


@pytest.mark.selenium  # noqa: C901
@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("job_id", list(range(num_jobs)))
def test_all_explainers_selenium(all_explanations, job_id):
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By
    from selenium import webdriver

    # Select explanations to target based on job id
    explanations = [
        explanation
        for i, explanation in enumerate(all_explanations)
        if i % num_jobs == job_id
    ]

    # Set up driver
    driver = webdriver.Firefox()
    driver.implicitly_wait(TIMEOUT)

    target_addr = ("127.0.0.1", 7100 + job_id)
    set_show_addr(target_addr)

    def goto_mini_url(explanation):
        mini_url = show_link(explanation)
        driver.get(mini_url)
        driver.find_element_by_class_name("card")

    def check_mini_overall_graph():
        # Expect overall graph
        wait.until(EC.presence_of_element_located((By.ID, "overall-graph--1")))

    def check_mini_specific_graph(index=0):
        # Click on specific graph
        dropdown_el = driver.find_element_by_class_name("Select-control")
        dropdown_el.click()
        specific_el = dropdown_el.find_element_by_xpath(
            "//div[contains(text(),'{} : ')]".format(index + 1)
        )
        specific_el.click()

        # Expect specific graph
        wait.until(EC.presence_of_element_located((By.ID, "graph-0-0")))

    # TODO: Investigate why this doesn't work in DevOps environment.
    # def check_close_specific_graph():
    #     # Click on close
    #     close_el = driver.find_element_by_class_name("Select-clear")
    #     close_el.click()
    #
    #     # Expect placeholder for select
    #     wait.until(EC.presence_of_element_located((By.CLASS_NAME, "Select-placeholder")))

    # Run mini app checks
    wait = WebDriverWait(driver, TIMEOUT)
    for explanation in explanations:
        goto_mini_url(explanation)
        check_mini_overall_graph()
        if explanation.selector is not None:
            check_mini_specific_graph(0)
            check_mini_specific_graph(1)
        # check_close_specific_graph()

    def goto_full_tab(explanation_type):
        tabs_el = driver.find_element_by_id("tabs")
        if explanation_type == "perf":
            tab_name = "Performance"
        else:
            tab_name = explanation_type.capitalize()
        tab_el = tabs_el.find_element_by_xpath(
            "//span[contains(text(),'{}')]".format(tab_name)
        )
        tab_el.click()

    def duplicate_explanations(explanation, num_duplicates):
        explanations = []
        for i in range(num_duplicates):
            an_explanation = deepcopy(explanation)
            an_explanation.name = "{}_{}".format(explanation.name, i)
            explanations.append(an_explanation)
        return explanations

    def goto_full_url(explanations, share_tables=True):
        full_url = show_link(explanations, share_tables=share_tables)
        driver.get(full_url)
        driver.find_element_by_id("overview-tab")

        wait.until(
            EC.text_to_be_present_in_element(
                (By.ID, "overview-tab"), "Welcome to Interpret ML"
            )
        )

    def check_full_dropdown(explanations):
        wait.until(
            EC.text_to_be_present_in_element(
                (By.CLASS_NAME, "card-title"), "Select Explanation"
            )
        )
        # Select items
        for explanation in explanations:
            dropdown_el = driver.find_element_by_class_name("Select-control")
            dropdown_el.click()

            item_el = dropdown_el.find_element_by_xpath(
                "//div[contains(text(),'{}')]".format(explanation.name)
            )
            item_el.click()

    def check_full_overall_graphs(explanations):
        if explanations[0].visualize() is None:
            return

        for i in range(len(explanations)):
            wait.until(
                EC.presence_of_element_located((By.ID, "overall-graph-{}".format(i)))
            )

    def check_full_specific_graphs(explanations, share_tables, num_records=2):
        if explanations[0].visualize(0) is None:
            return

        if share_tables:
            # Expect shared container
            wait.until(
                EC.presence_of_element_located(
                    (By.XPATH, "//div[contains(text(),'Select Components to Graph')]")
                )
            )

            # Click on records
            for i in range(num_records):
                record_path = "(//input[@type='checkbox'])[{}]".format(i + 1)
                record_el = driver.find_element_by_xpath(record_path)
                record_el.click()
        else:
            # Expect multiple containers
            for i in range(len(explanations)):
                wait.until(
                    EC.presence_of_element_located(
                        (
                            By.XPATH,
                            "//div[@class='gr']/div[@class='gr-col'][{}]".format(i + 1),
                        )
                    )
                )

            # Click on records
            for explanation_idx in range(len(explanations)):
                for record_idx in range(num_records):
                    record_path = "(//div[@class='gr-col'][{}]//input[@type='checkbox'])[{}]".format(
                        explanation_idx + 1, record_idx + 1
                    )
                    record_el = driver.find_element_by_xpath(record_path)
                    record_el.click()

        # Expect specific graphs
        for explanation_idx in range(len(explanations)):
            for record_idx in range(num_records):
                wait.until(
                    EC.presence_of_element_located(
                        (By.ID, "graph-{}-{}".format(explanation_idx, record_idx))
                    )
                )

    # Run full app checks
    for explanation in explanations:
        for share_tables in [True, False]:
            # NOTE: Suspected loading issue around cytoscape. Investigate later.
            if isinstance(explanation, TreeExplanation):
                continue

            explanations = duplicate_explanations(explanation, 2)
            goto_full_url(explanations, share_tables=share_tables)
            goto_full_tab(explanations[0].explanation_type)
            check_full_dropdown(explanations)
            check_full_overall_graphs(explanations)
            check_full_specific_graphs(explanations, share_tables)

    driver.close()
    shutdown_show_server()
