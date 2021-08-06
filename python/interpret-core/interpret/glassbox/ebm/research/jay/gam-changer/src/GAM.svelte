
<script>
  import ContGlobalExplain from './global-explanation/ContFeature.svelte';
  import CatGlobalExplain from './global-explanation/CatFeature.svelte';
  import InterContCatGlobalExplain from './global-explanation/InterContCatFeature.svelte';
  import InterContContGlobalExplain from './global-explanation/InterContContFeature.svelte';
  import InterCatCatGlobalExplain from './global-explanation/InterCatCatFeature.svelte';
  import Sidebar from './sidebar/Sidebar.svelte';
  import ToggleSwitch from './components/ToggleSwitch.svelte';
  import Dropzone from './components/Dropzone.svelte';

  import * as d3 from 'd3';
  import { initEBM } from './ebm';
  import { initDummyEBM} from './dummyEbm';
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import { downloadJSON, round } from './utils/utils';
  import { getBinEdgeScore } from './utils/ebm-edit';

  import redoIconSVG from './img/redo-icon.svg';
  import undoIconSVG from './img/undo-icon.svg';
  import exportIconSVG from './img/export-icon.svg';

  export let modelName = null;
  export let sampleName = null;

  let data = null;
  let sampleData = null;
  let isClassification = null;
  let ebm = initDummyEBM(0, 0, 0, 0);
  let component = null;
  let toggle = null;
  let changer = null;
  let featureSelect = null;
  let selectedFeature = null;
  let updateChanger = true;
  let featureSelectList = null;

  // Some view size constants
  const svgHeight = 500;
  const svgWidth = svgHeight / 2 * 3;
  const sidebarWidth = 250;

  const confusionMatrixData = {
    tn: [23.2, null, 23, null],
    fn: [23.2, null, 23, null],
    fp: [23.2, null, 23, null],
    tp: [23.2, null, 23, null]
  };

  // [original, last, current, last last]
  const barData = {
    accuracy: [0.5, null, 0.5, null],
    rocAuc: [0.5, null, 0.5, null],
    balancedAccuracy: [0.5, null, 0.5, null]
  };

  // Create stores to pass to child components
  let historyList = null;
  let historyStore = writable([]);
  historyStore.subscribe(value => {historyList = value;});

  let sidebarInfo = {};
  let sidebarStore = writable({
    rmse: 0,
    mae: 0,
    accuracy: 0,
    rocAuc: 0,
    balancedAccuracy: 0,
    confusionMatrix: [],
    prCurve: [],
    rocCurve: [],
    curGroup: '',
    selectedTab: 'effect',
    effectScope: 'global',
    historyHead: 0,
    previewHistory: false,
    barData: JSON.parse(JSON.stringify(barData)),
    confusionMatrixData: JSON.parse(JSON.stringify(confusionMatrixData)),
    totalSampleNum: 0
  });

  let footerStore = writable({
    sample: '',
    slice: '',
    help: '',
    state: '',
    baselineInit: false,
    baseline: 0,
  });

  let footerActionStore = writable('');

  sidebarStore.subscribe(async value => {
    sidebarInfo = value;

    // Listen to sampleDataCreated and featureDataCreated (user uploads file)
    if (value.curGroup === 'sampleDataCreated') {
      sampleData = value.loadedData;
      value.loadedData = null;
      value.curGroup = '';

      // Initialize the sidebar view
      await initSidebar();

      sidebarStore.set(sidebarInfo);
    }

    if (value.curGroup === 'modelDataCreated') {
      data = value.loadedData;
      value.loadedData = null;
      value.curGroup = '';

      // Initialize the GAM View
      await initGAMView();

      // If the user has already loaded sampleData, we init sidebar here too
      if (sampleData !== null) await initSidebar();

      sidebarStore.set(sidebarInfo);
    }

    // User can also directly upload a .gamchanger file
    if (value.curGroup === 'gamchangerCreated') {
      data = value.loadedData.modelData;
      sampleData = value.loadedData.sampleData;

      // Also restore the history stack
      historyList = value.loadedData.historyList;
      historyStore.set(historyList);

      value.loadedData = null;
      value.curGroup = '';

      // Initialize the GAM View & sidebar view
      await initGAMView();
      await initSidebar();

      sidebarStore.set(sidebarInfo);
    }

    // Double check to make sure the head matches the current feature
    if (value.curGroup === 'headChanged') {
      const headFeatureName = historyList[value.historyHead].featureName;
      if (headFeatureName !== selectedFeature.name) {
        // Search the feature name in all types
        const types = ['continuous', 'categorical', 'interaction'];
        types.forEach(t => {
          let curIndex = featureSelectList[t].findIndex(d => d.name === headFeatureName);
          if (curIndex !== -1) {
            selectedFeature = {};
            selectedFeature.type = t;
            selectedFeature.data = data.features[featureSelectList.continuous[curIndex].featureID];
            selectedFeature.id = featureSelectList.continuous[curIndex].featureID;
            selectedFeature.name = featureSelectList.continuous[curIndex].name;
            featureSelect.selectedIndex = curIndex;
            
            sidebarInfo.featureName = headFeatureName;
            ebm.setEditingFeature(headFeatureName);

            resizeFeatureSelect();
            updateChanger = !updateChanger;

            sidebarStore.set(sidebarInfo);
          }
        });
      }
    }
  });

  // Bind the SVGs
  const preProcessSVG = (svgString) => {
    return svgString.replaceAll('black', 'currentcolor')
      .replaceAll('fill:none', 'fill:currentcolor')
      .replaceAll('stroke:none', 'fill:currentcolor');
  };

  /**
   * Dynamically bind SVG files as inline SVG strings in this component
   */
  const bindInlineSVG = () => {
    d3.select(component)
      .selectAll('.svg-icon.icon-redo')
      .html(preProcessSVG(redoIconSVG));

    d3.select(component)
      .selectAll('.svg-icon.icon-undo')
      .html(preProcessSVG(undoIconSVG));

    d3.select(component)
      .selectAll('.svg-icon.icon-export')
      .html(preProcessSVG(exportIconSVG));
  };

  /**
   * Initialize the GAM view
   * This function assumes the variable `data` is not null
   */
  const initGAMView = async () => {
    if (data === null) return;

    isClassification = data.isClassifier;

    // Create a list of feature select options (grouped by types, sorted by importance)
    featureSelectList = {
      continuous: [],
      categorical: [],
      interaction: []
    };

    data.features.forEach((f, i) => {
      featureSelectList[f.type].push({
        name: f.name,
        featureID: i,
        importance: f.importance
      });
    });

    // Sort each feature type by alphabetical order
    Object.keys(featureSelectList).forEach(k => featureSelectList[k].sort((a, b) => a.name.localeCompare(b.name)));

    // Populate the slice option list
    let selectElement = d3.select(component).select('#feature-select');
    
    // Remove existing options
    selectElement.selectAll('option').remove();

    let featureGroups = ['continuous', 'categorical', 'interaction'];

    featureGroups.forEach(type => {
      let groupName = type.charAt(0).toUpperCase() + type.slice(1);
      let optGroup = selectElement.append('optgroup')
        .attr('label', groupName + ' (name - importance)');
      
      featureSelectList[type].forEach(opt => {
        optGroup.append('option')
          .attr('value', opt.featureID)
          .attr('data-level', opt.level)
          .text(`${opt.name} - ${round(opt.importance, 3)}`);
      });
    });

    // Use the latest edited feature as the initial feature is history is restored
    let targetFeatureIndex = null;
    let tempSelectedFeature = {};

    if (historyList.length > 0) {
      const lastEditName = historyList[historyList.length - 1].featureName;
      data.features.forEach(f => {
        if (f.name === lastEditName) {
          tempSelectedFeature.type = f.type;
        }
      });
      targetFeatureIndex = featureSelectList[tempSelectedFeature.type].map(d => d.name).indexOf(lastEditName);
    } else {
      // Initialize GAM Changer using the continuous variable with the highest importance

      targetFeatureIndex = d3.maxIndex(featureSelectList.continuous, d => d.importance);
      tempSelectedFeature.type = 'continuous';

      // targetFeatureIndex = d3.maxIndex(featureSelectList.categorical, d => d.importance);
      // tempSelectedFeature.type = 'categorical';
    }
    
    tempSelectedFeature.data = data.features[featureSelectList[tempSelectedFeature.type][targetFeatureIndex].featureID];
    tempSelectedFeature.id = featureSelectList[tempSelectedFeature.type][targetFeatureIndex].featureID;
    tempSelectedFeature.name = featureSelectList[tempSelectedFeature.type][targetFeatureIndex].name;

    // featureSelect has a different index system from the featureSelectList
    // console.log(Array.from(featureSelect.options));
    let selectElementTargetIndex = Array.from(featureSelect.options).reduce((a, d, i) => {
      if (parseInt(d.value) === tempSelectedFeature.id) a.push(i);
      return a;
    }, [])[0];
    featureSelect.selectedIndex = selectElementTargetIndex;

    resizeFeatureSelect();

    // Need to set the name for EBM (even if it is a dummy EBM) so that
    // ContFeature can start drawing
    ebm.setEditingFeature(tempSelectedFeature.name);

    // If we are reconstructing from .gamchanger file, we can load the real EBM here
    // and re-do all the changes in the history
    if (historyList.length > 0) {
      ebm = await initEBM(data, sampleData, historyList[0].featureName, isClassification);

      // Get the initial metrics
      let metrics = ebm.getMetrics();

      if (ebm.isClassification) {
        sidebarInfo.accuracy = metrics.accuracy;
        sidebarInfo.rocAuc = metrics.rocAuc;
        sidebarInfo.balancedAccuracy = metrics.balancedAccuracy;
        sidebarInfo.confusionMatrix = metrics.confusionMatrix;
      } else {
        sidebarInfo.rmse = metrics.rmse;
        sidebarInfo.mae = metrics.mae;
      }

      for (let i = 0; i < historyList.length; i++) {
        let curCommit = historyList[i];
        if (curCommit.featureName !== ebm.editingFeatureName) {
          ebm.setEditingFeature(curCommit.featureName);
        }
        if (curCommit.type !== 'original') {
          let result = getBinEdgeScore(curCommit.state.pointData);
          ebm.setModel(result.newBinEdges, result.newScores);
        }
      }

      sidebarInfo.historyHead = historyList.length - 1;
    }

    selectedFeature = tempSelectedFeature;
    updateChanger = !updateChanger;

    sidebarInfo.totalSampleNum = 0;
    sidebarInfo.featureName = selectedFeature.name;
    sidebarInfo.historyHead = 0;
    sidebarInfo.previewHistory = false;

    sidebarStore.set(sidebarInfo);

    footerStore.update(value => {
      value.totalSampleNum = sidebarInfo.totalSampleNum;
      value.sample = `<b>0/${sidebarInfo.totalSampleNum }</b> test samples selected`;
      return value;
    });
  };

  /**
   * Initialize the sidebar view
   * This function assumes that both `data` and `sampleData` are loaded
   */
  const initSidebar = async () => {
    if (data === null || sampleData === null) return;

    isClassification = data.isClassifier;

    // Create the sidebar feature data
    let featurePlotData = {cont: [], cat: []};
    let featureDataNameMap = new Map();
    data.features.forEach((d, i) => featureDataNameMap.set(d.name, i));

    let sampleDataNameMap = new Map();
    sampleData.featureNames.forEach((d, i) => sampleDataNameMap.set(d, i));

    // Initialize an EBM object
    if (ebm.isDummy !== undefined) {
      ebm = await initEBM(data, sampleData, selectedFeature.name, isClassification);

      // Get the initial metrics
      let metrics = ebm.getMetrics();

      if (ebm.isClassification) {
        sidebarInfo.accuracy = metrics.accuracy;
        sidebarInfo.rocAuc = metrics.rocAuc;
        sidebarInfo.balancedAccuracy = metrics.balancedAccuracy;
        sidebarInfo.confusionMatrix = metrics.confusionMatrix;
      } else {
        sidebarInfo.rmse = metrics.rmse;
        sidebarInfo.mae = metrics.mae;
      }
    }

    // Set curgroup outside the if block => metrics might be initialized form
    // initGamView() when a .gamchanger file is given
    sidebarInfo.curGroup = 'original';

    // Get the distribution of test data on each variable
    const testDataHistCount = ebm.getHistBinCounts();

    for (let j = 0; j < testDataHistCount.length; j++) {
      let curName = sampleData.featureNames[j];
      let curType = sampleData.featureTypes[j];

      if (curType === 'continuous') {
        let histEdge = data.features[featureDataNameMap.get(curName)].histEdge.slice(0, -1);
        featurePlotData.cont.push({
          id: sampleDataNameMap.get(curName),
          name: curName,
          histEdge: histEdge,
          histCount: testDataHistCount[j],
          histSelectedCount: new Array(testDataHistCount[j].length).fill(0)
        });
      } else {
        let histEdge = data.features[featureDataNameMap.get(curName)].histEdge;
        featurePlotData.cat.push({
          id: sampleDataNameMap.get(curName),
          name: curName,
          histEdge: histEdge,
          histCount: testDataHistCount[j],
          histSelectedCount: new Array(testDataHistCount[j].length).fill(0)
        });
      }
    }

    sidebarInfo.featurePlotData = featurePlotData;

    // Remember the number of total samples
    sidebarInfo.totalSampleNum = sampleData.samples.length;
    footerStore.update(value => {
      value.totalSampleNum = sidebarInfo.totalSampleNum;
      value.sample = `<b>0/${sidebarInfo.totalSampleNum }</b> test samples selected`;
      return value;
    });

    // Get the list of all categorical variables and their values to popularize
    // the slice select dropdown
    let sliceOptions = [];
    data.features.forEach(f => {
      if (f.type === 'categorical') {
        let curOptionGroup = [];
        f.binLabel.forEach(b => curOptionGroup.push(
          {
            name: f.name,
            level: b,
            levelName: data.labelEncoder[f.name][parseInt(b)],
            featureID: sampleDataNameMap.get(f.name)
          }
        ));
        sliceOptions.push(curOptionGroup);
      }
    });
    
    sliceOptions.sort((a, b) => a[0].name.localeCompare(b[0].name));
    sidebarInfo.sliceOptions = sliceOptions;
    
    sidebarStore.set(sidebarInfo);

    // If the GAM View is initialized, then the 0 item in history stack has
    // metrics from the dummy EBM. Need to update it here
    if (historyList.length > 0) {
      while (sidebarInfo.curGroup !== 'originalCompleted') {
        await new Promise(r => setTimeout(r, 500));
      }
      historyList[0].metrics.barData = JSON.parse(JSON.stringify(sidebarInfo.barData));
      historyList[0].metrics.confusionMatrixData = JSON.parse(JSON.stringify(sidebarInfo.confusionMatrixData));
      historyStore.set(historyList);
    }

  };

  const initData = async (modelName='iow-house-ebm-binary', sampleName='iow-house-sample-binary') => {
    console.log('loading data');
    isClassification = true;
    // data = await d3.json('/data/iow-house-ebm-binary.json');
    // data = await d3.json('/data/mimic2-model.json');
    // data = await d3.json('/data/iow-house-ebm.json');
    // data = await d3.json('/data/medical-ebm.json');
    data = await d3.json(`/data/${modelName}.json`);
    console.log(data);

    // sampleData = await d3.json('/data/iow-house-sample-binary.json');
    // sampleData = await d3.json('/data/mimic2-sample-1000.json');
    // sampleData = await d3.json('/data/medical-ebm-sample.json');
    sampleData = await d3.json(`/data/${sampleName}.json`);
    console.log(sampleData);
    console.log('loaded data');

    initGAMView();
    initSidebar();
  };

  /**
   * Wrapper to call the child changer's handler
  */
  const selectModeSwitched = () => {
    changer.selectModeSwitched();
  };

  const footerActionTriggered = (message) => {
    footerActionStore.set(message);

    if (message === 'save') {

      // Check if the user has confirmed all edits
      let allReviewed = true;
      historyList.forEach(d => allReviewed = allReviewed && d.reviewed);
      if (allReviewed) {
        let fileName = new Date().toLocaleDateString().replaceAll('/', '-');
        fileName = `edit-${fileName}.gamchanger`;

        // Create the .gamchanger project file
        let downloadData = {
          modelData: data,
          sampleData: sampleData,
          historyList: historyList
        };

        downloadJSON(downloadData, d3.select(component).select('#download-anchor'), fileName);
      } else {
        alert('You need to confirm all edits in the History panel (click 👍 icons) before saving the model.');
      }
    }
  };

  /**
   * Change the width of the select button so it fits the current content
   */
  const resizeFeatureSelect = () => {
    let opt = featureSelect.options[featureSelect.selectedIndex];

    let hiddenSelect = d3.select(component)
      .select('#hidden-select')
      .style('display', 'initial');

    hiddenSelect.select('#hidden-option')
      .text(opt.text);
    
    let selectWidth = hiddenSelect.node().clientWidth + 5 + 'px';
    hiddenSelect.style('display', 'none');
      
    d3.select(component)
      .select('#feature-select')
      .style('width', selectWidth);
  };

  const featureChanged = async () => {
    console.log('feature select changed');
    resizeFeatureSelect();

    // Get the selected feature
    let opt = featureSelect.options[featureSelect.selectedIndex];

    let selectedFeatureID = opt.value;

    // Update the selected feature object
    const curFeatureData = data.features[selectedFeatureID];

    // If the selected feature is interaction, figure out which two types
    if (curFeatureData.type === 'interaction') {
      let twoTypes = curFeatureData.id.map(i => data.features[i].type);

      if (twoTypes.includes('continuous') &&  twoTypes.includes('categorical')) {
        selectedFeature.type = 'cont-cat';
      } else if (twoTypes.includes('continuous')) {
        selectedFeature.type = 'cont-cont';
      } else {
        selectedFeature.type = 'cat-cat';
      }
    } else {
      selectedFeature.type = curFeatureData.type;
    }

    selectedFeature.name = curFeatureData.name;
    selectedFeature.data = curFeatureData;
    selectedFeature.id = selectedFeatureID;

    // Update the ebm model
    // TODO: update the model for interaction term as well
    if (curFeatureData.type !== 'interaction') {
      console.log('Change feature in ebm');
      ebm.setEditingFeature(selectedFeature.name);
      ebm = ebm;
    }

    // Restore and recover the metrics (need this step other wise the metrics
    // is forgotten if user switches feature during selected tab mode)
    sidebarInfo.curGroup = 'overwrite';
    sidebarStore.set(sidebarInfo);
    while (sidebarInfo.curGroup !== 'overwriteCompleted') {
      await new Promise(r => setTimeout(r, 300));
    }

    // Update the metrics
    let metrics = ebm.getMetrics();

    if (ebm.isClassification) {
      sidebarInfo.accuracy = metrics.accuracy;
      sidebarInfo.rocAuc = metrics.rocAuc;
      sidebarInfo.balancedAccuracy = metrics.balancedAccuracy;
      sidebarInfo.confusionMatrix = metrics.confusionMatrix;
    } else {
      sidebarInfo.rmse = metrics.rmse;
      sidebarInfo.mae = metrics.mae;
    }

    sidebarInfo.curGroup = 'new-feature-original';

    // Force the effect scope to be 'global'
    sidebarInfo.effectScope = 'global';
    sidebarInfo.featureName = selectedFeature.name;

    sidebarStore.set(sidebarInfo);

    console.log(selectedFeature);

    // Make sure all the updates are done before calling the following code
    // (It would trigger a view update)
    updateChanger = !updateChanger;

    // Reset the toggle button
    toggle.reset();
  };

  // Bind command-z and command-shift-z for undo and redo
  // Bind command-a as select all
  const bindShortcutKey = (undoCallback, redoCallback, selectAllCallback) => {
    d3.select('body')
      .on('keydown', e => {
        if ((e.metaKey || e.ctrlKey) && !e.shiftKey && e.key === 'z') {
          e.preventDefault();
          e.stopPropagation();
          undoCallback();
        } else if ((e.metaKey && e.shiftKey && e.key === 'z') ||
          (e.ctrlKey && e.shiftKey && e.key === 'Z')) {
          e.preventDefault();
          e.stopPropagation();
          redoCallback();
        } else if ((e.metaKey || e.ctrlKey) && !e.shiftKey && e.key === 'a') {
          e.preventDefault();
          e.stopPropagation();
          selectAllCallback();
        }
      });
  };

  $: modelName && sampleName && initData(modelName, sampleName);

  onMount(() => {
    bindInlineSVG();
    bindShortcutKey(
      () => footerActionTriggered('undo'),
      () => footerActionTriggered('redo'),
      () => footerActionTriggered('selectAll')
    );
  });

</script>

<style type='text/scss'>

  @import 'define';

  .main-tool {
    display: flex;
    flex-direction: column;
    border: 1px solid $gray-border;
    border-radius: $my-border-radius;
    background: white;
  }

  .tool {
    display: flex;
    flex-direction: row;
  }

  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 10px;
    border-bottom: 1px solid $gray-border;
    background: white;
    border-top-left-radius: $my-border-radius;
    height: 53px;

    .header__info {
      display: flex;
      align-items: center;
    }

    .header__control-panel {
      display: flex;
      align-items: center;
    }

    .header__history {
      background: hsl(225, 53%, 93%);
      border-radius: $my-border-radius;
      padding: 1px 7px;
      font-size: 0.9em;
      color: $gray-900;
      margin-left: 1em;

      &.past {
        background: hsl(35.3, 100%, 90%);
      }
    }
  }

  .select {
    display: flex;
    flex-direction: column;
    justify-content: center;

    select {
      height: 2em;
      border-radius: $my-border-radius;
      padding-top: 0;
      padding-bottom: 0;
      padding-left: 10px;
      border: 1px solid hsl(0, 0%, 85.9%);
      background: hsl(0, 20%, 99%);

      &:hover {
        border: 1px solid hsl(0, 0%, 71%);
      }

      &:focus {
        box-shadow: none;
      }
    }
  }

  .select:not(.is-multiple):not(.is-loading)::after {
    border-color: $blue-dark;
    right: 12px;
  }

  .select select:not([multiple]) {
    padding-right: 30px;
  }

  #hidden-select {
    display: none;
  }

  .toggle-switch-wrapper {
    width: 180px;
  }

  .tool-footer {
    display: flex;
    border-top: 1px solid $gray-border;
    height: 2em;
    align-items: center;
    border-bottom-left-radius: $my-border-radius;
    border-bottom-right-radius: $my-border-radius;
    padding: 5px 0px 5px 10px;

  }

  .message-line {
    display: flex;
    gap: 5px;
    flex-grow: 1;
    font-size: 0.9em;
    height: 100%;
    text-overflow: ellipsis;
    white-space: nowrap;
    overflow: hidden;

    span {
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
    }
  }

  .button-group {
    display: flex;
  }

  .field {
    border-bottom-right-radius: $my-border-radius;
  }

  .button {
    padding: 3px 0.8em;
    height: auto;
    border-radius: 0;
    border-top-width: 0px;
    border-bottom-width: 0px;
    border-color: $gray-border;
    border: 0px;

    &:hover {
      background: $gray-200;
      border-color: $gray-border;
    }

    &.right-button {
      border-right: 0px;
      border-bottom-right-radius: $my-border-radius;
      padding-right: 1em;
    }

    &:focus:not(:active) {
      box-shadow: none;
    }
  }

  .svg-icon {
    height: 100%;
    color: $indigo-dark;
    display: inline-flex;
    align-items: center;

    :global(svg) {
      width: 0.9em;
      height: 0.9em;
    }
  }

  .separator {
    margin: 0 3px;
    width: 1px;
    background-color: $gray-border;
    height: 100%;
    flex-shrink: 0;
  }

  .feature-window {
    background-color: $brown-50;
    border-radius: $my-border-radius;
  }

</style>

<div class='main-tool' bind:this={component} style={`width: ${sidebarWidth + svgWidth + 2}px;`}>
  <a id="download-anchor" style="display:none"> </a>

  <div class='tool'>
    <div class='feature-window'>

      <div class='header'>

        <div class='header__info'>

          <div class='select'>
            <select name='feature'
              bind:this={featureSelect}
              id='feature-select'
              title='Select a feature'
              on:blur={() => {}}
              on:change={featureChanged}
            >
              <option>No data loaded</option>
            </select>
          </div>

          <div class='select'>
            <select id='hidden-select'>
              <option id='hidden-option'></option>
            </select>
          </div>

          {#if selectedFeature !== null}
            <div class='header__history' class:past={sidebarInfo.previewHistory}>
              <span class='hash'>
                {#if sidebarInfo.historyHead === 0}
                  Original
                {:else}
                  {#if sidebarInfo.previewHistory}
                    Previous Edit:
                  {:else}
                    Latest Edit:
                  {/if}
                  {historyList[sidebarInfo.historyHead].hash.substring(0, 7)}
                {/if}
              </span>
            </div>
          {/if}

        </div>

        <div class='header__control-panel'>
          <div class='toggle-switch-wrapper'>
            <ToggleSwitch name='cont' bind:this={toggle}
              on:selectModeSwitched={selectModeSwitched}
            />
          </div>
        </div>

      </div>


      {#key updateChanger}

        {#if selectedFeature!== null}
        
          {#if selectedFeature.type === 'continuous'}
            <ContGlobalExplain
              featureData = {selectedFeature === null ? null : selectedFeature.data}
              scoreRange = {data === null ? null : data.scoreRange}
              bind:ebm = {ebm}
              bind:this = {changer}
              sidebarStore = {sidebarStore}
              footerStore = {footerStore}
              footerActionStore = {footerActionStore}
              historyStore = {historyStore}
              svgHeight = 500
            />
          {/if}

          {#if selectedFeature.type === 'categorical'}
            <CatGlobalExplain
              featureData = {selectedFeature === null ? null : selectedFeature.data}
              labelEncoder = {data === null ? null : data.labelEncoder[selectedFeature.name]}
              scoreRange = {data === null ? null : data.scoreRange}
              svgHeight = 500
              bind:this = {changer}
            />
          {/if}

          {#if selectedFeature.type === 'cont-cont'}
            <InterContContGlobalExplain
              featureData = {selectedFeature === null ? null : selectedFeature.data}
              scoreRange = {data === null ? null : data.scoreRange}
              svgHeight = 500
              bind:this = {changer}
            />
          {/if}

          {#if selectedFeature.type === 'cont-cat'}
            <InterContCatGlobalExplain
              featureData = {selectedFeature === null ? null : selectedFeature.data}
              labelEncoder = {data === null ? null : data.labelEncoder}
              scoreRange = {data === null ? null : data.scoreRange}
              svgHeight = 500
              chartType = 'bar'
              bind:this = {changer}
            />
          {/if}

          {#if selectedFeature.type === 'cat-cat'}
            <InterCatCatGlobalExplain
              featureData = {selectedFeature === null ? null : selectedFeature.data}
              labelEncoder = {data === null ? null : data.labelEncoder}
              scoreRange = {data === null ? null : data.scoreRange}
              svgHeight = 500
              bind:this = {changer}
            />
          {/if}

        {:else}
          <!-- If the feature is not loaded, we show the dropzone -->
          <div class='dropzone-wrapper' style={`width: ${svgWidth}px; height: ${svgHeight + 6}px;`}>
            <Dropzone sidebarStore={sidebarStore} dataType={'modelData'}/>
          </div>
        {/if}

      {/key}

    </div>

    <div class='sidebar-wrapper' style={`width: ${sidebarWidth}px;`}>
      <Sidebar sidebarStore={sidebarStore}
        historyStore={historyStore}
        width={sidebarWidth}
        sampleDataInitialized={sampleData !== null}
        ebm={ebm}
      />
    </div>
  </div>

  <div class='tool-footer'>

    <div class='message-line'>
      {#if selectedFeature !== null}
        <span>{@html $footerStore.help}</span>
        <div class='separator'></div>

        <span>{@html $footerStore.sample}</span>
        <span>{@html $footerStore.slice}</span>
        <div class='separator'></div>

        <span>{@html $footerStore.state}</span>
      {/if}
    </div>

      
    <div class='field has-addons'>

      <div class='control'>
        <button class='button' title='undo last edit' on:click={() => footerActionTriggered('undo')}>
          <span class='icon is-small'>
            <div class='svg-icon icon-undo'></div>
          </span>
        </button>
      </div>

      <div class='control'>
        <button class='button' title='redo last undo' on:click={() => footerActionTriggered('redo')}>
          <span class='icon is-small'>
            <div class='svg-icon icon-redo'></div>
          </span>
        </button>
      </div>

      <div class='control'>
        <button class='button right-button' title='save edits' on:click={() => footerActionTriggered('save')}>
          <span class='icon is-small'>
            <div class='svg-icon icon-export'></div>
          </span>
        </button>
      </div>

    </div>

  </div>

</div>