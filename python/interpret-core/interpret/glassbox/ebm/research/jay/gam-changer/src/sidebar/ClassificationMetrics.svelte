<script>
  import * as d3 from 'd3';
  import { onMount } from 'svelte';
  import { drawClassificationBarChart, drawConfusionMatrix } from './draw-metric';

  export let sidebarStore;

  let component = null;
  let sliceSelect = null;
  let loadingBar = null;
  let sidebarInfo = {};
  let width = 0;
  let height = 0;

  const svgPadding = {top: 10, right: 20, bottom: 40, left: 20};

  let confusionMatrixData = {
    tn: [23.2, null, 23, null],
    fn: [23.2, null, 23, null],
    fp: [23.2, null, 23, null],
    tp: [23.2, null, 23, null]
  };

  // [original, last, current, last last]
  let barData = {
    accuracy: [0.5, null, 0.5, null],
    rocAuc: [0.5, null, 0.5, null],
    balancedAccuracy: [0.5, null, 0.5, null]
  };

  onMount(() => {
    width = component.getBoundingClientRect().width;
    height = component.getBoundingClientRect().height;

    let svgInstance = d3.select(component)
      .select('.bar-svg')
      .node()
      .parentNode;
    
    let scrollBarWidth = svgInstance.offsetWidth - svgInstance.clientWidth;

    console.log(`effect tab: [${width}, ${height}, ${scrollBarWidth}]`);

    width -= scrollBarWidth;

    // Initialize the size of all svgs
    d3.select(component)
      .selectAll('.bar-svg')
      .attr('width', width)
      .attr('height', 455);

    // Expose loading bar to other components
    sidebarInfo.loadingBar = loadingBar;
    d3.select(loadingBar).classed('animated', false);
    sidebarStore.set(sidebarInfo);
  });

  const copyMetricData = (barData, confusionMatrixData, fromIndex, toIndex) => {
    barData.accuracy[toIndex] = barData.accuracy[fromIndex];
    barData.rocAuc[toIndex] = barData.rocAuc[fromIndex];
    barData.balancedAccuracy[toIndex] = barData.balancedAccuracy[fromIndex];

    confusionMatrixData.tn[toIndex] = confusionMatrixData.tn[fromIndex]; 
    confusionMatrixData.fn[toIndex] = confusionMatrixData.fn[fromIndex];
    confusionMatrixData.fp[toIndex] = confusionMatrixData.fp[fromIndex];
    confusionMatrixData.tp[toIndex] = confusionMatrixData.tp[fromIndex];
  };

  const tabClicked = (tab) => {
    let opt = sliceSelect.options[sliceSelect.selectedIndex];

    if (sidebarInfo.effectScope !== tab) {
      if (tab !== 'slice') {
        sidebarInfo.curGroup = `${tab}Clicked`;
        sidebarInfo.effectScope = tab;
        sidebarStore.set(sidebarInfo);
      } else {
        if (opt.value !== 'slice') {
          sidebarInfo.curGroup = `${tab}Clicked`;
          sidebarInfo.effectScope = tab;
          sidebarStore.set(sidebarInfo);
        }
      }
    }
  };

  const sliceChanged = () => {
    // Change the focus to the slice tab
    sidebarInfo.curGroup = 'sliceClicked';
    sidebarInfo.effectScope = 'slice';

    // Get the selected variable info
    let opt = sliceSelect.options[sliceSelect.selectedIndex];

    // User clicks the default 'slice' => jump back to global
    if (opt.value === 'slice') {
      sidebarInfo.curGroup = 'globalClicked';
      sidebarInfo.effectScope = 'global';
    } else {
      sidebarInfo.sliceInfo = {
        featureID: parseInt(opt.value),
        level: parseInt(opt.dataset.level)
      };
    }
    sidebarStore.set(sidebarInfo);
  };

  /**
   * Load the data from sidebarInfo to barData, confusionMatrixData at index i
   */
  const updateData = (index, barData, confusionMatrixData, sidebarInfo) => {
    if (sidebarInfo.confusionMatrix.length === 0) {
      return;
    }
    barData.accuracy[index] = sidebarInfo.accuracy;
    barData.rocAuc[index] = sidebarInfo.rocAuc;
    barData.balancedAccuracy[index] = sidebarInfo.balancedAccuracy;

    let total = sidebarInfo.confusionMatrix.reduce((a, b) => a + b);

    confusionMatrixData.tn[index] = sidebarInfo.confusionMatrix[0] / total;
    confusionMatrixData.fn[index] = sidebarInfo.confusionMatrix[1] / total;
    confusionMatrixData.fp[index] = sidebarInfo.confusionMatrix[2] / total;
    confusionMatrixData.tp[index] = sidebarInfo.confusionMatrix[3] / total;
  };

  sidebarStore.subscribe(value => {
    sidebarInfo = value;

    switch(sidebarInfo.curGroup) {
    case 'original': {
      updateData(0, barData, confusionMatrixData, sidebarInfo);

      copyMetricData(barData, confusionMatrixData, 0, 2);

      sidebarInfo.barData = JSON.parse(JSON.stringify(barData));
      sidebarInfo.confusionMatrixData = JSON.parse(JSON.stringify(confusionMatrixData));
      sidebarInfo.curGroup = 'originalCompleted';

      // Popularize the slice option list
      let selectElement = d3.select(component).select('#slice-select');
      sidebarInfo.sliceOptions.forEach(optionGroup => {
        let optGroup = selectElement.append('optgroup')
          .attr('label', optionGroup[0].name);
        
        optionGroup.forEach(opt => {
          optGroup.append('option')
            .attr('value', opt.featureID)
            .attr('data-level', opt.level)
            .text(opt.levelName);
        });
      });

      sidebarStore.set(sidebarInfo);
      break;
    }

    case 'new-feature-original': {
      // Nullify the last metric
      Object.keys(barData).forEach(k => barData[k][1] = null);
      Object.keys(confusionMatrixData).forEach(k => confusionMatrixData[k][1] = null);

      sidebarInfo.barData = JSON.parse(JSON.stringify(barData));
      sidebarInfo.confusionMatrixData = JSON.parse(JSON.stringify(confusionMatrixData));
      sidebarInfo.curGroup = 'originalCompleted';

      sidebarStore.set(sidebarInfo);

      break;
    }

    case 'current':
      updateData(2, barData, confusionMatrixData, sidebarInfo);
      break;

    case 'last':
      // Copy current to last
      copyMetricData(barData, confusionMatrixData, 2, 1);
      break;

    case 'commit': 

      // Copy last to last last
      copyMetricData(barData, confusionMatrixData, 1, 3);

      // Track the barData and confusionMatrix in the store so they can saved
      // in the history stack (only if the commit is global)
      if (sidebarInfo.effectScope === 'global') {
        sidebarInfo.barData = JSON.parse(JSON.stringify(barData));
        sidebarInfo.confusionMatrixData = JSON.parse(JSON.stringify(confusionMatrixData));
        
        sidebarInfo.curGroup = 'commitCompleted';
        sidebarStore.set(sidebarInfo);
      }

      break;
    
    case 'commit-not-global': {
      let globalBarData = sidebarInfo.barData;
      let globalConfusionMatrixData = sidebarInfo.confusionMatrixData;

      // Copy last to last last
      copyMetricData(globalBarData, globalConfusionMatrixData, 1, 3);

      // Copy current to last
      copyMetricData(globalBarData, globalConfusionMatrixData, 2, 1);

      // Update the current 
      updateData(2, globalBarData, globalConfusionMatrixData, sidebarInfo);

      // Save into the history stack
      sidebarInfo.barData = JSON.parse(JSON.stringify(globalBarData));
      sidebarInfo.confusionMatrixData = JSON.parse(JSON.stringify(globalConfusionMatrixData));

      sidebarInfo.curGroup = 'commitCompleted';
      sidebarStore.set(sidebarInfo);

      break;
    }

    case 'recover':
      // Copy last to current, copy last last to last
      copyMetricData(barData, confusionMatrixData, 1, 2);
      copyMetricData(barData, confusionMatrixData, 3, 1);
      break;

    case 'overwrite':
      barData = JSON.parse(JSON.stringify(sidebarInfo.barData));
      confusionMatrixData = JSON.parse(JSON.stringify(sidebarInfo.confusionMatrixData));

      sidebarInfo.curGroup = 'overwriteCompleted';
      sidebarStore.set(sidebarInfo);
      break;

    case 'nullify':
      confusionMatrixData = {
        tn: [null, null, null, null],
        fn: [null, null, null, null],
        fp: [null, null, null, null],
        tp: [null, null, null, null]
      };

      barData = {
        accuracy: [null, null, null, null],
        rocAuc: [null, null, null, null],
        balancedAccuracy: [null, null, null, null]
      };

      sidebarInfo.curGroup = 'nullifyCompleted';
      sidebarStore.set(sidebarInfo);
      break;

    case 'nullify-last':
      Object.keys(confusionMatrixData).forEach(k => confusionMatrixData[k][1] = null);
      Object.keys(barData).forEach(k => barData[k][1] = null);
      break;

    case 'original-only':
      updateData(0, barData, confusionMatrixData, sidebarInfo);
      break;

    case 'last-only':
      updateData(1, barData, confusionMatrixData, sidebarInfo);
      break;

    case 'current-only':
      updateData(2, barData, confusionMatrixData, sidebarInfo);
      break;
    
    default:
      break;
    }

    // We only update the graph if the user is currently in this tab
    if (sidebarInfo.selectedTab === 'effect') {
      drawClassificationBarChart(width, svgPadding, component, barData);
      drawConfusionMatrix(width, svgPadding, component, confusionMatrixData); 
    }

  });

</script>

<style type='text/scss'>

  @import '../define';

  .metrics-tab {
    height: 100%;
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    overflow-y: hidden;
    overflow-x: hidden;
  }

  .metrics {
    height: 100%;
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    overflow-y: scroll;
    overflow-x: hidden;
    position: relative;
    background-color: $brown-50;
    border-top: 1px solid $blue-50;

    svg {
      flex-shrink: 0;
    }
  }

  .text {
    width: 100%;
    text-align: center;
  }

  .bar-svg {
    margin-top: 0px;
  }

  .scope-selection.field {
    margin-top: 11px;
    margin-bottom: 11px;
    padding: 0 5px;
    justify-content: space-around;
  }

  .control {
    flex: 1;
  }

  .button {
    width: 100%;
    padding: 1px 11px;
    font-size: 0.9em;
    font-weight: 400;
    height: auto;
    border-color: $gray-border;

    &:hover {
      background: $gray-100;
    }

    &:focus:not(:active) {
      box-shadow: none;
    }

    &:focus {
      border-color: $gray-border;
    }

    &.selected {
      background: $gray-200;
    }
  }

  .select {
    text-overflow: ellipsis;
    width: 100%;
  }

  :global(.metrics-tab text.bar) {
    fill: hsl(230, 100%, 11%);
    font-size: 13px;
  }

  :global(.metrics-tab rect.original) {
    fill: $gray-300;
  }

  :global(.metrics-tab rect.last) {
    fill: $pastel1-orange;
  }

  :global(.metrics-tab rect.current) {
    fill: $pastel1-blue;
  }

  :global(.metrics-tab svg text) {
    cursor: default;
  }

  :global(.metrics-tab .metric-title) {
    dominant-baseline: hanging;
    font-size: 1.1em;
    font-weight: 600;
  }

  :global(.metrics-tab .legend-title) {
    dominant-baseline: hanging;
    text-anchor: middle;
    font-size: 0.8em;
    font-weight: 300;
    fill: $indigo-dark;
  }

  :global(.metrics-tab .bar-label) {
    dominant-baseline: middle;
    text-anchor: start;
    font-size: 0.8em;
    font-weight: 200;
    fill: $indigo-dark;
  }

  :global(.metrics-tab .matrix-label) {
    dominant-baseline: middle;
    text-anchor: middle;
    font-size: 0.8em;
    font-weight: 300;
    fill: $indigo-dark;
  }

  :global(.metrics-tab .matrix-explanation) {
    dominant-baseline: hanging;
    text-anchor: start;
    font-size: 0.7em;
    font-weight: 300;
    fill: hsl(0, 0%, 45%);
  }

  :global(.metrics-tab .dominant-middle) {
    dominant-baseline: middle;
  }
  

</style>

<div class='metrics-tab' bind:this={component}>

    <div class='scope-selection field has-addons'>

      <div class='control'>
        <button class='button' title='Show model performance across all test samples'
          class:selected={sidebarInfo.effectScope === 'global'}
          on:click={() => tabClicked('global')}
        >
          <span class='is-small'>
            Global
          </span>
        </button>
      </div>

      <div class='control'>
        <button class='button' title='Show model performance on the selected test samples'
          class:selected={sidebarInfo.effectScope === 'selected'}
          on:click={() => tabClicked('selected')}
        >
          <span class='is-small'>
            Selected
          </span>
        </button>
      </div>

      <div class='control'>
        <select class='button right-button select' name='slice'
          bind:this={sliceSelect}
          id='slice-select'
          title='Show model performance on the sliced test samples'
          class:selected={sidebarInfo.effectScope === 'slice'}
          on:click={() => tabClicked('slice')}
          on:blur={() => {}}
          on:change={sliceChanged}
        >
          <option selected value='slice'>Slice</option>

          <!-- <span style='color: hsla(0, 0%, 0%, 0.2); margin-left: 5px;'>▼</span> -->
      </select>
      </div>

    </div>

  <div class='metrics'>
    <svg class='bar-svg'></svg>
  </div>

</div>