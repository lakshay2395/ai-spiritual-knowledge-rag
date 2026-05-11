document.getElementById('search-btn').addEventListener('click', performSearch);
document.getElementById('query-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') performSearch();
});

async function performSearch() {
    const query = document.getElementById('query-input').value.trim();
    const religion = document.querySelector('input[name="religion"]:checked').value;
    
    if (!query) return;

    // UI State: Loading
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');
    document.getElementById('search-btn').disabled = true;

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                religion: religion === 'both' ? null : religion,
                top_k: 10
            }),
        });

        if (!response.ok) {
            throw new Error('Seeking wisdom failed. Please try again.');
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        alert(error.message);
    } finally {
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('search-btn').disabled = false;
    }
}

function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    const answerText = document.getElementById('answer-text');
    const sourcesList = document.getElementById('sources-list');

    // Set Answer
    answerText.textContent = data.answer;

    // Set Sources
    sourcesList.innerHTML = '';
    data.sources.forEach(source => {
        const card = document.createElement('div');
        card.className = 'source-card';
        card.innerHTML = `
            <span class="citation">${source.citation}</span>
            <p class="snippet">"${source.text}"</p>
        `;
        sourcesList.appendChild(card);
    });

    resultsDiv.classList.remove('hidden');
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}
