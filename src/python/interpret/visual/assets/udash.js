/*
Copyright (c) 2019 Microsoft Corporation
Distributed under the MIT software license
*/

// NOTE: This fixes the table widths, which normally only updates on resize.
var targetNode = document;
var config = { attributes: true, childList: true, subtree: true };
var callback = function(mutationsList, observer) {
    for(var mutation of mutationsList) {
        const el = mutation.target;
        if (el.classList.contains('gr')) {
            window.dispatchEvent(new Event('resize'));
        }
    }
};
var observer = new MutationObserver(callback);
observer.observe(targetNode, config);
