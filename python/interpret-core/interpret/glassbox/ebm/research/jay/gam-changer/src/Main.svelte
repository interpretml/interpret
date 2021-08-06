<script>
  import Header from './Header.svelte';
  import GAM from './GAM.svelte';
  import Tooltip from './components/Tooltip.svelte';

  import { tooltipConfigStore } from './store';

  // Set up tooltip
  let tooltip = null;
  let tooltipConfig = null;
  tooltipConfigStore.subscribe(value => {tooltipConfig = value;});

</script>

<style type='text/scss'>

  @import 'define';
  
  .main-standalone {
    display: flex;
    flex-direction: column;
  }

  .content {
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    gap: 50px;
    padding: 30px 0 30px 0;
    justify-content: center;
    align-items: center;
    height: min(800px, calc(100vh - 50px));
    overflow-y: scroll;
    width: 100vw;
    box-sizing: border-box;
    border: 1px solid $gray-border;
  }

  .feature-window {
    border: 1px solid $gray-border;
    margin: auto 0;
    border-radius: 5px;
  }

</style>

<div class='main-standalone'>
  <div class='header'>
    <Header />
  </div>

  <Tooltip bind:this={tooltip}/>

  <div class='content'>

    <GAM />

  </div>
  

</div>