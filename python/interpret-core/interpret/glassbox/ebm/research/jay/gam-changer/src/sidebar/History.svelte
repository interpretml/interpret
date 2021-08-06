<script>
  import * as d3 from 'd3';
  import { onMount, beforeUpdate, afterUpdate } from 'svelte';
  import { slide, fade, crossfade } from 'svelte/transition';
  import { quadInOut, expoInOut, cubicInOut } from 'svelte/easing';
  import { round, shuffle } from '../utils/utils';
  import { bindInlineSVG } from '../utils/svg-icon-binding';
  
  export let sidebarStore;
  export let historyStore;
  export let ebm;

  let component = null;
  let historyList = [];
  let sidebarInfo = null;
  let needToBindSVGs = false;

  // Auto scroll
  let autoscroll = null;
  let historyDIV = null;

  const editTypeIconMap = {
    'increasing': 'icon-increasing',
    'decreasing': 'icon-decreasing',
    'move': 'icon-updown',
    'align': 'icon-merge',
    'equal-interpolate': 'icon-interpolation',
    'inplace-interpolate': 'icon-inplace',
    'equal-regression': 'icon-interpolation',
    'inplace-regression': 'icon-regression',
    'delete': 'icon-delete',
    'original': 'icon-original'
  };

  sidebarStore.subscribe(value => {
    sidebarInfo = value;
  });

  historyStore.subscribe(value => {
    historyList = value;
    needToBindSVGs = true;
  });

  /**
   * Overwrite the edge definition in the EBM WASM model.
   * @param {string} curGroup Message to the metrics sidebar
   * @param {object} curNodeData Node data in `state`
   * @param {featureName} featureName The name of feature to be edited
   */
  const setEBM = async (curGroup, curNodeData, featureName=undefined) => {

    // Update the complete bin edge definition in the EBM model
    let newBinEdges = [];
    let newScores = [];

    // The left point will always have index 0
    let curPoint = curNodeData[0];
    let curEBMID = 0;

    while (curPoint.rightPointID !== null) {
      // Collect x and y
      newBinEdges.push(curPoint.x);
      newScores.push(curPoint.y);

      // Update the new ID so we can map them to bin indexes later (needed for
      // selection to check sample number)
      curPoint.ebmID = curEBMID;
      curEBMID++;

      curPoint = curNodeData[curPoint.rightPointID];
    }

    // Add the right node
    newBinEdges.push(curPoint.x);
    newScores.push(curPoint.y);
    curPoint.ebmID = curEBMID;

    await ebm.setModel(newBinEdges, newScores, featureName);
  };

  /**
   * Format the millisecond time to string
   * @param time Time in millisecond
   */
  const getTime = (time) => {
    let date = new Date(time);
    let hour = date.getHours();
    let minute = String(date.getMinutes()).padStart(2, '0');
    let months = ['January','February','March','April','May','June','July',
      'August','September','October','November','December'].map(d => d.substring(0, 3));
    let month = months[date.getMonth()];
    let day = date.getDate();
    let year  = date.getFullYear();

    return `${hour}:${minute} ${month} ${day}, ${year}`;
  };

  const checkboxClicked = (i) => {
    historyList[i].reviewed = !historyList[i].reviewed;
    historyList = historyList;
  };

  const previewClicked = (i) => {
    sidebarStore.update(value => {
      if (value.historyHead !== i) {
        value.curGroup = 'headChanged';
      }

      value.previewHistory = i !== historyList.length - 1;
      value.historyLastHead = value.historyHead;
      value.historyHead = i;

      // Force the effect scope to be global
      value.effectScope = 'global';
      return value;
    });
  };

  const deleteClicked = async (i) => {

    // Search the history stack to see if it is the last commit on this feature
    const featureName = historyList[i].featureName;
    let isLastCommit = true;
    for (let j = i + 1; j < historyList.length; j++) {
      if (historyList[j].featureName === featureName) {
        isLastCommit = false;
        break;
      }
    }

    // Need user to confirm the delete action if it is not the last commit
    let result = true;
    if (!isLastCommit) {
      result = confirm(
        `Deleting this commit will also remove all later commits on feature ${featureName}. This action cannot be undone. Is it OK?`);
    } else {
      result = confirm('Deleting a commit cannot be undone. Is it OK?');
    }

    if (!result) {
      return;
    }

    // Iterate the end to i, delete all commits on this feature
    for (let j = historyList.length - 1; j >= i; j--) {
      if (historyList[j].featureName === featureName) {
        historyList.splice(j, 1);
      }
    }

    // Need to reset EBM (fallback to the last commit on this feature)
    // We can always find a commit to fall back to because the first
    // original commit cannot never be deleted
    for (let j = i - 1; j >= 0; j--) {
      if (historyList[j].featureName === featureName) {
        await setEBM('current', historyList[j].state.pointData, featureName);
        break;
      }
    }


    // Update the HEAD if HEAD is at/after the deleted commit
    // If if the HEAD is on a different feature, the index is influenced
    if (sidebarInfo.historyHead >= i) {
      sidebarInfo.curGroup = 'headChanged';
      sidebarInfo.previewHistory = false;
      sidebarInfo.historyHead = historyList.length - 1;

      // Force the effect scope to be global
      sidebarInfo.effectScope = 'global';
    }

    historyStore.set(historyList);
    sidebarStore.set(sidebarInfo);

  };


  // beforeUpdate(() => {
  //   if (historyDIV) {
  //     console.log(historyDIV.offsetHeight, historyDIV.scrollTop, historyDIV.scrollHeight);
  //     console.log(historyDIV.offsetHeight + historyDIV.scrollTop > historyDIV.scrollHeight);
  //   }
  //   autoscroll = historyDIV &&
  //     (historyDIV.offsetHeight + historyDIV.scrollTop) > (historyDIV.scrollHeight );
  // });

  afterUpdate(() => {
    // if (autoscroll) {
    //   historyDIV.scrollTo(0, historyDIV.scrollHeight);
    // }

    if (needToBindSVGs) {
      bindInlineSVG(component);
      needToBindSVGs = false;
    }
  });

  onMount(() => {
    bindInlineSVG(component);
  });

</script>

<style type='text/scss'>

  @import '../define';

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

  .history-tab {
    height: 100%;
  }

  .history {
    height: 100%;
    width: 100%;
    overflow-y: scroll;
    overflow-x: hidden;
    position: relative;
    display: flex;
    flex-direction: column;
  }

  .commit {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    border-bottom: 1px solid $gray-border;
    padding: 5px 10px;
    font-size: 0.9em;
    gap: 5px;
    // opacity: 0.5;

    // &.current {
    //   opacity: 1;
    // }
  }

  .commit-title {
    width: 100%;
    display: flex;
    flex-direction: row;
    align-items: baseline;
    justify-content: space-between;
    cursor: default;
    gap: 5px;
  }

  .commit-feature {
    font-weight: 400;
    color: $gray-700;
    word-break: break-all;

    &.current {
      font-weight: 600;
      color: $blue-dark;
    }
  }

  .commit-time {
    font-size: 0.9em;
    color: $gray-700;
    flex-shrink: 0;
  }

  .commit-content {
    width: 100%;
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 0.5em;
    position: relative;
  }

  .commit-pen {
    position: absolute;
    right: 8px;
    bottom: 8px;

    .svg-icon {
      color: $gray-700;
      fill: $gray-700;
      opacity: 0.4;
      display: flex;

      :global(svg) {
        width: 0.9em;
        height: 0.9em;
      }
    }
  }

  .commit-message {
    width: 100%;
    background: $gray-100;
    padding: 5px 10px;
    border: 1px solid $gray-border;
    border-radius: 2px;

    .commit-message-text {
      margin-right: 0.2em;
    }

    &:focus {
      background: none;
    }
  }

  .commit-checkbox {
    // margin-left: auto;
    cursor: pointer;
    width: 1em;
    height: 1em;
  }

  @keyframes thumbup {
    0%{
      transform: scale(0);
      -webkit-transform: scale(0);
    }
    80%{
      transform: scale(1.3);
      -webkit-transform: scale(1.3);
    }
    100%{
      transform: scale(1);
      -webkit-transform: scale(1);
    }
  }

  .commit-icon {
    .svg-icon {
      color: $gray-700;
      fill: $gray-700;
      opacity: 0.6;
      display: flex;

      :global(svg) {
        width: 1.2em;
        height: 1.2em;
      }

      &.current {
        color: $blue-dark;
        fill: $blue-dark;
        opacity: 1;
      }
    }
  }

  .commit-bar {
    @extend .commit-title;
    justify-content: flex-end;
    align-items: flex-start;
    gap: 10px;

    .svg-icon {
      cursor: pointer;
      color: $indigo-dark;
      fill: $indigo-dark;

      :global(svg) {
        width: 1em;
        height: 1em;
      }

      &:hover {
        color: $blue-600;
        fill: $blue-600;
      }

      &.selected {
        color: $blue-reg;
        fill: $blue-reg;
      }

      &.disabled, &.disabled:hover {
        cursor: not-allowed;
        color: $gray-700;
        fill: $gray-700;
        opacity: 0.5;
      }

      &.disabled:active {
        pointer-events: none;
      }
    }

    .checkbox-box {
      &.hidden {
        display: none;
      }
    }

    .checkbox-check {
      animation: thumbup 150ms ease-in;
      -webkit-animation: thumbup 150ms ease-in;

      &.hidden {
        display: none;
      }

      .svg-icon {
        color: $blue-reg;
        fill: $blue-reg;
      }
    }
  }

  .commit-hash {
    font-family: ui-monospace,SFMono-Regular,SF Mono,Consolas,Liberation Mono,Menlo,monospace;
    font-size: 0.9em;
    color: $gray-700;
    margin-right: auto;
    cursor: copy;

    &:hover {
      color: $blue-600;
    }
  }

  .hidden {
    display: none;
  }

  .empty-history {
    display: flex;
    height: 100%;

    span {
      margin: auto;
    }

    &.hidden {
      display: none;
    }
  }

</style>

<div class='history-tab' bind:this={component}>

  <div class='icon-loading'>
  </div>

  <div class='empty-history' class:hidden={historyList.length > 0}>
    <span>No GAM is changed :(</span>
  </div>

  <div class='history' bind:this={historyDIV}>

    {#each historyList as history, i}

      <div class='commit' class:current={sidebarInfo.featureName === history.featureName}
        transition:slide={{duration: 300}}
      >
        <!-- Header -->
        <div class='commit-title'>

          <div class='commit-feature' class:current={sidebarInfo.featureName === history.featureName}>
            {history.featureName}
          </div>

          <div class='commit-time'>
            {getTime(history.time)}
          </div>

        </div>

        <!-- Content -->
        <div class='commit-content'>

          <div class='commit-icon'>
            <div class={`svg-icon ${editTypeIconMap[history.type]}`}
              class:current={sidebarInfo.featureName === history.featureName}
              title={history.type}
            ></div>
          </div>

          <span class='commit-message' contenteditable bind:innerHTML={history.description}>
          </span>

          {#if i === 0}
            <div class='commit-pen'>
              <div class='svg-icon icon-pen'></div>
            </div>
          {/if}

        </div>

        <!-- Footer -->
        <div class='commit-bar'>
          <div class='commit-hash' title={'copy hash: ' + history.hash}
            on:click={() => navigator.clipboard.writeText(history.hash)}
          >
            {history.hash.substring(0, 7)}
            {#if sidebarInfo.historyHead === i}
              {' (HEAD)'}
            {/if}
          </div>

          <div class='commit-checkbox' title='confirm'
            on:click={() => checkboxClicked(i)}
          >
            <div class='checkbox-box' class:hidden={history.reviewed}>
              <div class='svg-icon icon-box'></div>
            </div>

            <div class='checkbox-check' class:hidden={!history.reviewed}>
              <div class='svg-icon icon-thumbup'></div>
            </div>
          </div>

          <div class='svg-icon icon-location selected'
            class:hidden={sidebarInfo.historyHead !== i}
            title='current'
            style='cursor: default;'
          ></div>

          <div class='svg-icon icon-right-arrow'
            class:hidden={sidebarInfo.historyHead === i}
            class:disabled={sidebarInfo.featureName !== history.featureName}
            title='check out'
            on:click={() => previewClicked(i)}
          ></div>

          <div class='svg-icon icon-commit-delete' title='delete'
            class:disabled={history.type === 'original'}
            on:click={() => deleteClicked(i)}
          ></div>

        </div>

      </div>

    {/each}

  </div>

</div>