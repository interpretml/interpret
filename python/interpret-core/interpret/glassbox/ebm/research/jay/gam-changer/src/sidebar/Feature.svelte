<script>

  import * as d3 from 'd3';
  import { onMount, afterUpdate } from 'svelte';
  import { flip } from 'svelte/animate';
  import { initLegend, initContFeature, initCatFeature, updateContFeature, updateCatFeature } from './draw-feature';

  export let sidebarStore;
  export let width = 0;

  let component = null;
  let selectedTab = 'cont';
  let sidebarInfo = {};
  let tabHeights = {cont: '100%', cat: '100%'};
  $: curTabHeight = tabHeights[selectedTab];
  let featureInitialized = false;
  let waitingToDrawDIV = false;
  
  let height = 0;

  const svgHeight = 40;

  const svgCatPadding = {top: 2, bottom: 2, left: 10, right: 10};
  const svgContPadding = svgCatPadding;

  const titleHeight = 10;
  let catBarWidth = 0;

  let sortedContFeatures = [];
  let sortedCatFeatures = [];


  onMount(() => {
    // width = component.getBoundingClientRect().width;
    height = component.getBoundingClientRect().height;

    let instance = d3.select(component)
      .select('.feature-list')
      .node();
    
    let scrollBarWidth = instance.offsetWidth - instance.clientWidth;

    console.log(`effect tab: [${width}, ${height}, ${scrollBarWidth}]`);

    width -= scrollBarWidth;
  });

  afterUpdate(() => {
    const totalSampleNum = sidebarInfo.totalSampleNum;

    if (waitingToDrawDIV) {

      // Draw the legend
      initLegend(component, width, svgCatPadding);

      sortedContFeatures.forEach(f => {
        initContFeature(component, f, svgContPadding, totalSampleNum,
          width, svgHeight, titleHeight);
      });

      // Find the max equal bar width
      catBarWidth = (width - svgCatPadding.left - svgCatPadding.right) /
        d3.max(sortedCatFeatures, d => d.histCount.length);

      sortedCatFeatures.forEach(f => {
        initCatFeature(component, f, svgCatPadding, totalSampleNum, width,
          catBarWidth, svgHeight, titleHeight);
      });

      waitingToDrawDIV = false;
    }

  });

  sidebarStore.subscribe(value => {
    sidebarInfo = value;

    // Initialize the feature elements in DOM when we have the data
    if (sidebarInfo.featurePlotData !== undefined && !featureInitialized) {
      featureInitialized = true;

      sortedContFeatures = sidebarInfo.featurePlotData.cont;
      sortedCatFeatures = sidebarInfo.featurePlotData.cat;

      // Wait for the DOM to update (will trigger afterUpdate)
      waitingToDrawDIV = true;
    }

    // Update the feature graph
    if (sidebarInfo.curGroup === 'updateFeature' && featureInitialized) {

      let tempSortedContFeatures = sidebarInfo.featurePlotData.cont;
      let tempSortedCatFeatures = sidebarInfo.featurePlotData.cat;

      // Update the overlay histogram
      const selectedSampleCount = sortedContFeatures[0].histSelectedCount.reduce((a, b) => a + b);
      const totalSampleCount = sortedContFeatures[0].histCount.reduce((a, b) => a + b);
      let needToResort = false;

      // Step 1: update the continuous feature graph
      tempSortedContFeatures.forEach(f => {
        needToResort = updateContFeature(component, f, selectedSampleCount,
          totalSampleCount, svgHeight, svgContPadding, titleHeight);
      });

      if (needToResort) {
        d3.timeout(() => {
          tempSortedContFeatures.sort((a, b) => b.distanceScore - a.distanceScore);
          sortedContFeatures = tempSortedContFeatures;
        }, 700);
      } else {
        sortedContFeatures = tempSortedContFeatures;
      }

      // Step 2: update the categorical feature graph
      needToResort = false;

      tempSortedCatFeatures.forEach(f => {
        needToResort = updateCatFeature(component, f, selectedSampleCount,
          totalSampleCount, svgHeight, svgCatPadding, titleHeight);
      });

      if (needToResort) {
        d3.timeout(() => {
          tempSortedCatFeatures.sort((a, b) => b.distanceScore - a.distanceScore);
          sortedCatFeatures = tempSortedCatFeatures;
        }, 700);
      } else {
        sortedCatFeatures = tempSortedCatFeatures;
      }
      
      sidebarInfo.curGroup = '';
      sidebarStore.set(sidebarInfo);
    }

  });

</script>

<style type='text/scss'>

  @import '../define';

  .feature-tab {
    height: 100%;
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .feature-list {
    height: 100%;
    width: 100%;
    overflow-y: scroll;
    overflow-x: hidden;
    position: relative;
    border-top: 1px solid $blue-50;
    background-color: $brown-50;
  }

  .feature-cont, .feature-cat {
    position: absolute;
    height: 100%;
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 0px;

    &.hidden {
      display: none;
    }
  }

  .feature-cont {
    .feature {
      border-bottom: 1px solid $pastel2-gray;
    }
  }

  .feature-legend {
    position: relative;
    padding: 5px 0 4px 0;
  }

  .feature {
    will-change: transform, height;
    margin: 0 auto;
  }

  .scope-selection.field {
    margin-top: 11px;
    margin-bottom: 11px;
  }

  .button {
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

    &:active {
      border-color: $gray-border;
    }

    &.selected {
      background: $gray-200;
    }
  }

  :global(.feature-tab .feature-title) {
    dominant-baseline: hanging;
    text-anchor: start;
    font-size: 0.7em;
    font-weight: 400;
    fill: black;

    &:global(.shadow) {
      stroke: white;
      fill: white;
    }
  }

  :global(.feature-tab .area-path) {
    fill: $pastel1-blue;
    stroke: $pastel1-blue;
    opacity: 1;
    stroke-linejoin: round;
  }

  :global(.feature-tab .global-bar) {
    fill: $pastel1-blue;
    opacity: 1;
  }

  :global(.feature-tab .selected-bar) {
    fill: $orange-400;
    opacity: 0.4;
  }

  :global(.feature-tab rect.selected-sample) {
    fill: $orange-400;
    opacity: 0.4;
  }

  :global(.feature-tab .legend-title) {
    dominant-baseline: hanging;
    text-anchor: middle;
    font-size: 0.8em;
    font-weight: 300;
    fill: $indigo-dark;
  }

</style>

<div class='feature-tab' bind:this={component}>

  <div class='scope-selection field has-addons'>

    <div class='control'>
      <button class='button' title='show continuous variables'
        class:selected={selectedTab === 'cont'}
        on:click={() => {selectedTab = 'cont';}}
      >
        <span class='is-small'>
          Continuous
        </span>
      </button>
    </div>

    <div class='control'>
      <button class='button' title='show categorical variables'
        class:selected={selectedTab === 'cat'}
        on:click={() => {selectedTab = 'cat';}}
      >
        <span class='is-small'>
          Categorical
        </span>
      </button>
    </div>

  </div>

  <div class='feature-list'>

    <div class='feature-legend'>
      <svg id='legend'></svg>
    </div>

    <div class='feature-cont' class:hidden={selectedTab !== 'cont'}>
      {#each sortedContFeatures as f (f.id)}
        <div class={`feature feature-${f.id}`}
          style={`height: ${svgHeight}px;`}
          animate:flip="{{duration: 800}}"
        >
          <svg id={`cont-feature-svg-${f.id}`} width={width} height={svgHeight}></svg>
        </div>
      {/each}
    </div>

    <div class='feature-cat' class:hidden={selectedTab !== 'cat'}>
      {#each sortedCatFeatures as f (f.id)}
        <div class={`feature feature-${f.id}`}
          style={`height: ${svgHeight}px;`}
          animate:flip="{{duration: 800}}"
        >
          <svg id={`cat-feature-svg-${f.id}`} width={width} height={svgHeight}></svg>
        </div>
      {/each}
    </div>

  </div>

</div>