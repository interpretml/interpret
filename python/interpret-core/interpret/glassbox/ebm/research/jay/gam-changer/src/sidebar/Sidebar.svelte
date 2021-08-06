<script>
  import { onMount } from 'svelte';
  import * as d3 from 'd3';

  import ClassificationMetrics from './ClassificationMetrics.svelte';
  import RegressionMetrics from './RegressionMetrics.svelte';
  import Dropzone from '../components/Dropzone.svelte';
  import Feature from './Feature.svelte';
  import History from './History.svelte';

  export let sidebarStore;
  export let historyStore;
  export let width;
  export let sampleDataInitialized;
  export let ebm;

  let sidebarInfo = {};
  let component = null;

  sidebarStore.subscribe(value => {
    sidebarInfo = value;
  });

  const updateSelectedTab = (newTab) => {
    sidebarInfo.selectedTab = newTab;
    sidebarInfo = sidebarInfo;
    sidebarStore.set(sidebarInfo);
  };

</script>

<style type='text/scss'>

  @import '../define';

  .sidebar {
    width: 100%;
    height: 100%;
    border-left: 1px double $gray-border;
    display: flex;
    flex-direction: column;
    background: white;
    border-top-right-radius: $my-border-radius;
    border-bottom-right-radius: $my-border-radius;
  }

  .header {
    height: 53px;
    border-bottom: 1px solid $gray-border;
    padding: 10px 0;
    border-top-right-radius: $my-border-radius;
    background: white;

    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    flex-shrink: 0;

    // Hack for firefox's space-evenly
    &::before, &::after {
      content: '';
    }
  }

  .tab-button {
    color: $gray-900;
    border-bottom: 2px solid white;
    border: 1px solid transparent;
    cursor: pointer;
    text-align: center;
    position: relative;

    display: inline-flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;

    &.selected {
      color: currentColor;
      font-weight: 600;

      &::before {
        content: '';
        position: absolute;
        width: 100%;
        top: 135%;
        left: 0;
        border-bottom: 4px solid $blue-600;
      }
    }

    &::after {
      content: attr(data-text);
      content: attr(data-text) / '';
      visibility: hidden;
      height: 0;
      pointer-events: none;
      overflow: hidden;
      font-weight: 600;
    }
  }

  .tab {
    height: 100%;
    width: 100%;
    position: absolute;
    top: 0;
    left: 0;

    &.hidden {
      visibility: hidden;
      pointer-events: none;
    }
  }

  .content {
    width: 100%;
    position: relative;
    flex-grow: 1;
    overflow-y: hidden;
    overflow-x: hidden;
  }


</style>

<div class='sidebar' bind:this={component} style={`width: ${width};`}>

  <div class='header'>
    <div class='tab-button'
      data-text='Effect'
      class:selected={sidebarInfo.selectedTab === 'effect'}
      on:click={() => {updateSelectedTab('effect');}}
    >
      Effect
    </div>

    <div class='tab-button'
      data-text='Feature'
      class:selected={sidebarInfo.selectedTab === 'feature'}
      on:click={() => {updateSelectedTab('feature');}}
    >
      Feature
    </div>

    <div class='tab-button'
      data-text='History'
      class:selected={sidebarInfo.selectedTab === 'history'}
      on:click={() => {updateSelectedTab('history');}}
    >
      History
    </div>

  </div>

  <div class='content'>

    <div class='tab' class:hidden={sidebarInfo.selectedTab !== 'effect'}>
      {#if sampleDataInitialized}
        <ClassificationMetrics sidebarStore={sidebarStore}/>
      {:else}
        <Dropzone sidebarStore={sidebarStore} dataType={'sampleData'}/>
      {/if}
    </div>

    <div class='tab' class:hidden={sidebarInfo.selectedTab !== 'feature'}>
      {#if sampleDataInitialized}
        <Feature sidebarStore={sidebarStore} width={width}/>
      {:else}
        <Dropzone sidebarStore={sidebarStore} dataType={'sampleData'}/>
      {/if}
    </div>

    <div class='tab' class:hidden={sidebarInfo.selectedTab !== 'history'}>
      <History sidebarStore={sidebarStore} historyStore={historyStore} ebm={ebm}/>
    </div>
    
  </div>

</div>