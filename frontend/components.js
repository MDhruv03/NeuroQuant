// Shared components for NeuroQuant frontend
// This ensures consistent header and footer across all pages

const NeuroQuantComponents = {
    // Render header with active page indicator
    renderHeader: function(activePage) {
        return `
        <header class="mono-card mx-4 mt-4 rounded-none p-6 border-2">
            <div class="container mx-auto flex justify-between items-center">
                <div class="flex items-center space-x-3">
                    <div class="w-10 h-10 border-2 border-white flex items-center justify-center">
                        <span class="text-white font-bold text-lg">NQ</span>
                    </div>
                    <h1 class="text-2xl font-light tracking-widest">NEUROQUANT</h1>
                </div>
                <nav>
                    <ul class="flex space-x-6">
                        <li>
                            <a href="index.html" class="${activePage === 'dashboard' ? 'text-white font-medium border-b-2 border-white pb-1' : 'text-gray-400 hover:text-white transition-colors font-light'}">
                                DASHBOARD
                            </a>
                        </li>
                        <li>
                            <a href="agents.html" class="${activePage === 'agents' ? 'text-white font-medium border-b-2 border-white pb-1' : 'text-gray-400 hover:text-white transition-colors font-light'}">
                                AGENTS
                            </a>
                        </li>
                        <li>
                            <a href="backtest.html" class="${activePage === 'history' ? 'text-white font-medium border-b-2 border-white pb-1' : 'text-gray-400 hover:text-white transition-colors font-light'}">
                                HISTORY
                            </a>
                        </li>
                    </ul>
                </nav>
            </div>
        </header>
        `;
    },

    // Render footer
    renderFooter: function() {
        return `
        <footer class="text-center py-8 mt-12">
            <p class="text-gray-500 font-light tracking-wide">© 2025 NEUROQUANT</p>
        </footer>
        `;
    },

    // Initialize components on page load
    init: function(activePage) {
        document.addEventListener('DOMContentLoaded', () => {
            // Inject header
            const headerContainer = document.getElementById('header-container');
            if (headerContainer) {
                headerContainer.innerHTML = this.renderHeader(activePage);
            }

            // Inject footer
            const footerContainer = document.getElementById('footer-container');
            if (footerContainer) {
                footerContainer.innerHTML = this.renderFooter();
            }
        });
    }
};
