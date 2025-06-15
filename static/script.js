const form = document.getElementById('analyze-form');
    const loader = document.getElementById('loader');
    const results = document.getElementById('results');
    const restartSection = document.getElementById('restart-section');
    const restartBtn = document.getElementById('restart-btn');
    const videoInput = document.getElementById('video-url');

    form.onsubmit = async (e) => {
        e.preventDefault();
        loader.style.display = 'block';
        results.style.display = 'none';
        restartSection.style.display = 'none';

        const url = videoInput.value;

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ video_url: url })
            });

            const data = await response.json();

            document.getElementById('count').textContent = data.count;
            document.getElementById('pos-count').textContent = data.positive;
            document.getElementById('neg-count').textContent = data.negative;
            document.getElementById('ratio').textContent = (data.ratio * 100).toFixed(1) + '%';
            document.getElementById('positive-comment').textContent = `"${data.most_positive.comment}" (Confidence: ${data.most_positive.confidence})`;
            document.getElementById('negative-comment').textContent = `"${data.most_negative.comment}" (Confidence: ${data.most_negative.confidence})`;

            loader.style.display = 'none';
            results.style.display = 'block';
            restartSection.style.display = 'block';
        } catch (error) {
            loader.innerHTML = '<p>❌ Error analyzing video. Please try again.</p>';
        }
    };

    restartBtn.onclick = () => {
        videoInput.value = '';
        results.style.display = 'none';
        restartSection.style.display = 'none';
        loader.style.display = 'none';
        videoInput.focus();
    };