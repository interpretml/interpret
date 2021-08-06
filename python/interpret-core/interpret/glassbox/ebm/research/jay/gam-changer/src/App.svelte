<svelte:head>
	<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bulma/0.9.1/css/bulma.min.css"
		integrity="sha512-ZRv40llEogRmoWgZwnsqke3HNzJ0kiI0+pcMgiz2bxO6Ew1DVBtWjVn0qjrXdT3+u+pSN36gLgmJiiQ3cQtyzA=="
		crossorigin="anonymous" />
</svelte:head>

<script>
	import Main from './Main.svelte';
	import Article from './Article.svelte';
</script>

<style type='text/scss'>
	.stand-alone-page {
		color: #333;
		box-sizing: border-box;
		font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen-Sans, Ubuntu, Cantarell, "Helvetica Neue", sans-serif;
		font-size: 16px;
		font-weight: 400;
		line-height: 1.5;
	}
</style>

<!-- Import Google font and font awesome -->
<link rel="preconnect" href="https://fonts.gstatic.com">
<link
		href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:ital,wght@0,200;0,300;0,400;0,600;0,700;0,900;1,200;1,300;1,400;1,600;1,700;1,900&display=swap"
		rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.13.1/css/all.min.css">

<div class='stand-alone-page'>
	<!-- <Main /> -->
	<Article />
</div>