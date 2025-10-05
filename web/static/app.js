

socket.on("metrics", (msg) => {
    status.textContent = "Metrics: " + JSON.stringify(msg.metrics);
    if (msg.cue) {
        document.getElementById("cue").textContent = msg.cue;
    }
});
