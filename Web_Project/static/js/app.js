// ML Web App - Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    console.log('ML Web App loaded');
    
    // Dark mode toggle
    initDarkMode();
    
    // Add loading overlay
    addLoadingOverlay();
    
    // File upload validation
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                // Check file size (16MB limit)
                const maxSize = 16 * 1024 * 1024;
                if (file.size > maxSize) {
                    alert('File size exceeds 16MB limit. Please upload a smaller file.');
                    fileInput.value = '';
                    return;
                }
                
                // Check file extension
                const fileName = file.name.toLowerCase();
                if (!fileName.endsWith('.csv')) {
                    alert('Please upload a CSV file.');
                    fileInput.value = '';
                    return;
                }
                
                console.log('File selected:', file.name, 'Size:', (file.size / 1024).toFixed(2), 'KB');
            }
        });
    }
    
    // Form submission loading state
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            showLoading();
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                const originalText = submitBtn.innerHTML;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
            }
        });
    });
    
    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            if (bsAlert) bsAlert.close();
        }, 5000);
    });
    
    // Smooth scroll
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Tooltip initialization
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

// Dark Mode Functions
function initDarkMode() {
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    
    // Add dark mode toggle button
    const toggleBtn = document.createElement('button');
    toggleBtn.className = 'theme-toggle';
    toggleBtn.innerHTML = '<i class="fas fa-' + (savedTheme === 'dark' ? 'sun' : 'moon') + ' fa-lg"></i>';
    toggleBtn.setAttribute('aria-label', 'Toggle dark mode');
    toggleBtn.addEventListener('click', toggleDarkMode);
    document.body.appendChild(toggleBtn);
}

function toggleDarkMode() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    
    // Update toggle button icon
    const toggleBtn = document.querySelector('.theme-toggle');
    if (toggleBtn) {
        toggleBtn.innerHTML = '<i class="fas fa-' + (newTheme === 'dark' ? 'sun' : 'moon') + ' fa-lg"></i>';
    }
}

// Loading Overlay Functions
function addLoadingOverlay() {
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.innerHTML = `
        <div class="spinner-container">
            <div class="spinner-border text-light" role="status"></div>
            <p class="mt-3">Processing your data...</p>
        </div>
    `;
    document.body.appendChild(overlay);
}

function showLoading() {
    const overlay = document.querySelector('.loading-overlay');
    if (overlay) overlay.classList.add('show');
}

function hideLoading() {
    const overlay = document.querySelector('.loading-overlay');
    if (overlay) overlay.classList.remove('show');
}

// Hide loading on page load
window.addEventListener('load', hideLoading);

// Table sorting function
function sortTable(table, columnIndex) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    const sorted = rows.sort((a, b) => {
        const aCell = a.cells[columnIndex].textContent.trim();
        const bCell = b.cells[columnIndex].textContent.trim();
        
        // Try numeric comparison
        const aNum = parseFloat(aCell);
        const bNum = parseFloat(bCell);
        
        if (!isNaN(aNum) && !isNaN(bNum)) {
            return aNum - bNum;
        }
        
        // String comparison
        return aCell.localeCompare(bCell);
    });
    
    // Remove and re-add rows
    rows.forEach(row => tbody.removeChild(row));
    sorted.forEach(row => tbody.appendChild(row));
}

// Image error handling
document.addEventListener('error', function(e) {
    if (e.target.tagName === 'IMG') {
        e.target.style.display = 'none';
    }
}, true);

// Download handlers
function downloadFile(url, filename) {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Chart interactions (for future enhancements)
function highlightChart(chartId) {
    const chart = document.getElementById(chartId);
    if (chart) {
        chart.style.border = '2px solid #0d6efd';
        setTimeout(() => {
            chart.style.border = '';
        }, 2000);
    }
}

// Copy to clipboard functionality
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showToast('Copied to clipboard!', 'success');
    }).catch(err => {
        console.error('Failed to copy:', err);
        showToast('Failed to copy', 'error');
    });
}

// Toast notification
function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) {
        const container = document.createElement('div');
        container.id = 'toastContainer';
        container.style.position = 'fixed';
        container.style.top = '20px';
        container.style.right = '20px';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
    }
    
    const toast = document.createElement('div');
    toast.className = `alert alert-${type} alert-dismissible fade show`;
    toast.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.getElementById('toastContainer').appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

// Export functionality
function exportResults(format) {
    console.log('Exporting results as', format);
    // Implementation depends on backend API
}

// Print results
function printResults() {
    window.print();
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + K: Focus search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.getElementById('search');
        if (searchInput) searchInput.focus();
    }
    
    // Escape: Close modals
    if (e.key === 'Escape') {
        const modals = document.querySelectorAll('.modal.show');
        modals.forEach(modal => {
            const bsModal = bootstrap.Modal.getInstance(modal);
            if (bsModal) bsModal.hide();
        });
    }
});

// Console welcome message
console.log('%cML Web App', 'font-size: 24px; font-weight: bold; color: #667eea;');
console.log('%cBuilt with Python, Flask, and scikit-learn', 'font-size: 12px; color: #6c757d;');
