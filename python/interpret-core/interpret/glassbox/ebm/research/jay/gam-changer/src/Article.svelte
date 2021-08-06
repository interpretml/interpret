<script>
  import * as d3 from 'd3';
  import { onMount } from 'svelte';
  import { fade, fly } from 'svelte/transition';

  import GAM from './GAM.svelte';
  import Tooltip from './components/Tooltip.svelte';

  import githubIconSVG from './img/github-icon.svg';
  import youtubeIconSVG from './img/youtube-icon.svg';
  import pdfIconSVG from './img/pdf-icon.svg';

  import { tooltipConfigStore } from './store';

  let component = null;

  // Set up tooltip
  let tooltip = null;
  let tooltipConfig = null;
  tooltipConfigStore.subscribe(value => {tooltipConfig = value;});

  // Set up tab interactions
  let tab = 'iowa';
  let tabNames = {
    iowa: {
      modelName: 'iow-house-ebm-binary',
      sampleName: 'iow-house-train-sample'
    },
    adult: {
      modelName: 'adult-model',
      sampleName: 'adult-sample'
    },
    my: {
      modelName: null,
      sampleName: null
    }
  };

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
      .selectAll('.svg-icon.icon-github')
      .html(preProcessSVG(githubIconSVG));

    d3.select(component)
      .selectAll('.svg-icon.icon-youtube')
      .html(preProcessSVG(youtubeIconSVG));

    d3.select(component)
      .selectAll('.svg-icon.icon-pdf')
      .html(preProcessSVG(pdfIconSVG));
  };

  const tabClicked = (newTab) => {
    tab = newTab;
  };

  onMount(() => {
    bindInlineSVG();
  });

</script>

<style type='text/scss'>

  @import 'define';
  
  .page {
    display: flex;
    flex-direction: column;
  }

  .top {
    position: relative;
    display: grid;
    height: min(850px, calc(100vh - 5px));
    grid-template-columns: [start] 1fr [mid-start] auto [mid-end] 1fr [end];
    grid-template-rows: [start] 1fr [content-start] auto [content-end] 1fr [end];
  }

  .top-fill {
    background: $blue-dark;
    grid-column: start / end;
    grid-row: start / end;
    height: 61.8%;
  }

  .top-empty {
    grid-column: end / start;
    grid-row: start / end;
    height: 30%;
  }

  .logo-container {
    grid-column: mid-start / mid-end;
    grid-row: start / content-start;
    display: flex;
    align-items: center;
    margin-left: 20px;
  }

  .logo-icon {
    height: auto;
    width: 100%;
    max-width: 400px;
  }

  .gam-changer {
    grid-column: mid-start / mid-end;
    grid-row: content-start / content-end;

    background: white;
    border-radius: 10px;
    box-shadow: 0px 10px 40px hsla(0, 0%, 0%, 0.2);
  }

  .icon-container {
    grid-column: mid-end / end;
    grid-row: content-start / content-end;
    width: 95px;

    display: flex;
    flex-direction: column;
    margin: 20px 0 0 25px;
    gap: 10px;

    a {
      display: flex;
      flex-direction: row;
      align-items: center;
      color: white;
      gap: 10px;

      span {
        font-size: 1.2em;
      }
    }
  }

  .svg-icon {
    height: 100%;
    color: white;
    display: inline-flex;
    align-items: center;

    :global(svg) {
      width: 2em;
      height: 2em;
    }
  }

  .gam-tab {
    grid-column: mid-start / mid-end;
    grid-row: content-end / end;

    display: flex;
    flex-direction: column;
    padding-top: 20px;
    width: 100%;

    .tab-title {
      text-align: center;
      font-size: 0.9em;
      font-variant: initial;
      color: gray;
      cursor: default;
    }

    .tab-options {
      display: flex;
      justify-content: center;
      align-items: flex-end;
      color: $blue-dark;
      gap: 15px;
      font-variant: small-caps;
      font-size: 1.1em;

      span.option {
        cursor: pointer;
        display: inline-flex;
        flex-direction: column;
        align-items: center;
        justify-content: space-between;

        &::after {
          content: attr(data-text);
          content: attr(data-text) / "";
          height: 0;
          visibility: hidden;
          overflow: hidden;
          user-select: none;
          pointer-events: none;
          font-weight: 600;
        }
      }

      span.selected {
        text-decoration: underline;
        font-weight: 600;
      }
    }
  }

</style>

<div class='page' bind:this={component}>
  <Tooltip bind:this={tooltip}/>

  <div class='top'>

    <div class='top-fill'>

    </div>

    <div class='top-empty'>

    </div>

    <div class='logo-container'>
      <div class='logo-icon'>
        <img src='/img/logo.svg' alt='logo img'>
      </div>
    </div>

    {#key tab}
      <div class='gam-changer' in:fly={{ x: 2000, duration: 800 }} out:fly={{ x : -2000, duration: 800 }}>
        <GAM modelName={tabNames[tab].modelName} sampleName={tabNames[tab].sampleName}/>
      </div>
    {/key}

    <div class='icon-container'>
      <a target="_blank" href="https://interpret.ml">
        <div class="svg-icon icon-github" title="Open-source code">
        </div>
        <span>Code</span>
      </a>

      <a target="_blank" href="https://interpret.ml">
        <div class="svg-icon icon-youtube" title="Open-source code">
        </div>
        <span>Video</span>
      </a>

      <a target="_blank" href="https://interpret.ml">
        <div class="svg-icon icon-pdf" title="Open-source code">
        </div>
        <span>Paper</span>
      </a>

    </div>

    <div class='gam-tab'>
      <!-- <span class='tab-title'>
        Choose a model
      </span> -->

      <div class='tab-options'>
        <span class='tab-title'>Choose a model:</span>
        <span class='option' class:selected={tab === 'iowa'}
          data-text='iowa house price'
          on:click={() => tabClicked('iowa')}>
          iowa house price
        </span>

        <span class='option' class:selected={tab === 'adult'}
          data-text='census income'
          on:click={() => tabClicked('adult')}>
          census income
        </span>

        <span class='option' class:selected={tab === 'my'}
          data-text='my model'
          on:click={() => tabClicked('my')}>
          my model
        </span>
      </div>
    </div>

  </div>



  <!-- <div class='content'>

    <h1>What is GAM Changer?</h1>

  </div> -->
  

</div>