/**
 * RAG Chat Interface JavaScript
 * Handles chat functionality with streaming support using existing API endpoints
 */

class ChatInterface {
    constructor() {
        console.log('ChatInterface: Initializing...');
        this.apiBaseUrl = window.location.origin;
        this.isConnected = true; // Start with connection enabled
        this.isStreaming = false;
        this.currentStreamController = null;
        
        this.initializeElements();
        this.attachEventListeners();
        
        // Set initial connection status to ready
        this.setConnectionStatus(true, 'Ready');
        
        // Check health in background (non-blocking)
        this.checkApiHealth();
        console.log('ChatInterface: Initialization complete');
    }

    initializeElements() {
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.statusDot = document.getElementById('status-dot');
        this.statusText = document.getElementById('status-text');
        this.streamingToggle = document.getElementById('streamingToggle');
        this.topKSelect = document.getElementById('topKSelect');
        this.similarityThreshold = document.getElementById('similarity-threshold');
        this.thresholdValue = document.getElementById('threshold-value');
    }

    attachEventListeners() {
        // Send button click
        this.sendButton.addEventListener('click', () => this.sendMessage());
        
        // Enter key to send (Shift+Enter for new line)
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => {
            this.autoResizeTextarea();
            this.updateSendButton();
        });

        // Similarity threshold slider
        if (this.similarityThreshold && this.thresholdValue) {
            this.similarityThreshold.addEventListener('input', (e) => {
                this.thresholdValue.textContent = e.target.value;
            });
        }

        // Initial button state
        this.updateSendButton();
    }

    autoResizeTextarea() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }

    updateSendButton() {
        const hasText = this.messageInput.value.trim().length > 0;
        const shouldDisable = !hasText || this.isStreaming || !this.isConnected;
        
        console.log('ChatInterface: updateSendButton called');
        console.log('ChatInterface: hasText =', hasText);
        console.log('ChatInterface: isStreaming =', this.isStreaming);
        console.log('ChatInterface: isConnected =', this.isConnected);
        console.log('ChatInterface: shouldDisable =', shouldDisable);
        
        this.sendButton.disabled = shouldDisable;
        console.log('ChatInterface: button.disabled =', this.sendButton.disabled);
    }

    async checkApiHealth() {
        try {
            console.log('ChatInterface: Checking API health...');
            
            // Add timeout to prevent hanging
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
            
            const response = await fetch(`${this.apiBaseUrl}/health`, {
                signal: controller.signal,
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                }
            });
            
            clearTimeout(timeoutId);
            
            if (response.ok) {
                const data = await response.json();
                console.log('ChatInterface: Health response:', response.status, data);
                
                if (data.status === 'healthy') {
                    this.setConnectionStatus(true, 'Connected');
                } else {
                    console.log('ChatInterface: API reports unhealthy, but allowing connection');
                    this.setConnectionStatus(true, 'API Ready');
                }
            } else {
                console.log('ChatInterface: Health endpoint returned error, but allowing connection');
                this.setConnectionStatus(true, 'API Ready');
            }
        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('ChatInterface: Health check timed out, but allowing connection');
            } else {
                console.error('ChatInterface: Health check failed:', error);
            }
            // Always allow the interface to work regardless of health check result
            this.setConnectionStatus(true, 'Connection Ready');
        }
    }

    setConnectionStatus(connected, message) {
        console.log('ChatInterface: setConnectionStatus called', connected, message);
        this.isConnected = connected;
        this.statusDot.className = `status-dot ${connected ? 'connected' : ''}`;
        this.statusText.textContent = message;
        console.log('ChatInterface: calling updateSendButton from setConnectionStatus');
        this.updateSendButton();
    }

    async sendMessage() {
        console.log('ChatInterface: sendMessage called');
        const message = this.messageInput.value.trim();
        console.log('ChatInterface: message =', message);
        console.log('ChatInterface: isStreaming =', this.isStreaming);
        console.log('ChatInterface: isConnected =', this.isConnected);
        
        if (!message || this.isStreaming || !this.isConnected) {
            console.log('ChatInterface: sendMessage blocked - conditions not met');
            return;
        }

        console.log('ChatInterface: Sending message...');
        
        // Add user message to chat
        this.addMessage('user', message);
        
        // Clear input and disable
        this.messageInput.value = '';
        this.autoResizeTextarea();
        this.isStreaming = true;
        this.updateSendButton();

        const useStreaming = this.streamingToggle.checked;
        const topK = parseInt(this.topKSelect.value);

        try {
            if (useStreaming) {
                await this.sendStreamingMessage(message, topK);
            } else {
                await this.sendRegularMessage(message, topK);
            }
        } catch (error) {
            console.error('Send message error:', error);
            this.addErrorMessage('Failed to send message. Please try again.');
        } finally {
            this.isStreaming = false;
            this.updateSendButton();
        }
    }

    async sendRegularMessage(message, topK) {
        try {
            const similarityThreshold = this.getSimilarityThreshold();
            const response = await fetch(`${this.apiBaseUrl}/ask?question=${encodeURIComponent(message)}&top_k=${topK}&similarity_threshold=${similarityThreshold}&include_meta=1`);
            const data = await response.json();

            if (response.ok) {
                if (data.error) {
                    this.addErrorMessage(data.error);
                } else {
                    this.addMessage('assistant', data.answer, data.sources);
                    
                    // Add context information if available
                    if (data.context_used !== undefined) {
                        const contextInfo = data.context_used
                            ? `Used ${data.relevant_documents_count} relevant documents (threshold: ${data.similarity_threshold})`
                            : `No relevant documents found (threshold: ${data.similarity_threshold}). Using general knowledge.`;
                        this.addContextInfo(contextInfo);
                    }
                }
            } else {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
        } catch (error) {
            console.error('Regular message error:', error);
            this.addErrorMessage('Failed to get response from the API.');
        }
    }

    async sendStreamingMessage(message, topK) {
        // Add typing indicator
        const typingElement = this.addTypingIndicator();
        let assistantMessageElement = null;
        let contentElement = null;
        let sourcesData = null;

        try {
            const response = await fetch(
                `${this.apiBaseUrl}/ask/stream?question=${encodeURIComponent(message)}&top_k=${topK}`,
                {
                    method: 'GET',
                    headers: {
                        'Accept': 'text/event-stream',
                        'Cache-Control': 'no-cache'
                    }
                }
            );

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line in buffer

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            
                            switch (data.type) {
                                case 'metadata':
                                    // Remove typing indicator and prepare for response
                                    if (typingElement) {
                                        typingElement.remove();
                                    }
                                    break;

                                case 'sources':
                                    sourcesData = data.sources;
                                    break;

                                case 'answer_start':
                                    // Create assistant message element
                                    assistantMessageElement = this.createMessageElement('assistant');
                                    contentElement = assistantMessageElement.querySelector('.message-content');
                                    this.chatMessages.appendChild(assistantMessageElement);
                                    this.scrollToBottom();
                                    break;

                                case 'answer_chunk':
                                    if (contentElement && data.chunk) {
                                        contentElement.textContent += data.chunk;
                                        this.scrollToBottom();
                                    }
                                    break;

                                case 'answer_complete':
                                    // Clean and format content, separating think blocks
                                    if (contentElement) {
                                        const { main, think } = splitThinkBlocks(contentElement.textContent);
                                        contentElement.innerHTML = formatMessageContent(main);
                                        if (think) {
                                            const thinkDiv = document.createElement('div');
                                            thinkDiv.className = 'think';
                                            thinkDiv.textContent = think;
                                            contentElement.appendChild(thinkDiv);
                                        }
                                    }
                                    // Add sources if available
                                    if (sourcesData && assistantMessageElement) {
                                        this.addSourcesToMessage(assistantMessageElement, sourcesData);
                                    }
                                    break;

                                case 'error':
                                    if (typingElement) {
                                        typingElement.remove();
                                    }
                                    this.addErrorMessage(data.message || 'An error occurred during streaming.');
                                    return;
                            }
                        } catch (parseError) {
                            console.error('Error parsing SSE data:', parseError, line);
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Streaming error:', error);
            if (typingElement) {
                typingElement.remove();
            }
            this.addErrorMessage('Streaming connection failed. Please try again.');
        }
    }

    addMessage(role, content, sources = null) {
        const messageElement = this.createMessageElement(role, content);
        
        if (sources && sources.length > 0) {
            this.addSourcesToMessage(messageElement, sources);
        }
        
        this.chatMessages.appendChild(messageElement);
        this.scrollToBottom();
        return messageElement;
    }

    createMessageElement(role, content = '') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (content) {
            const { main, think } = splitThinkBlocks(content);
            contentDiv.innerHTML = formatMessageContent(main);
            if (think) {
                const thinkDiv = document.createElement('div');
                thinkDiv.className = 'think';
                thinkDiv.textContent = think;
                contentDiv.appendChild(thinkDiv);
            }
        }
        
        const timestampDiv = document.createElement('div');
        timestampDiv.className = 'message-timestamp';
        timestampDiv.textContent = new Date().toLocaleTimeString();
        
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(timestampDiv);
        
        return messageDiv;
    }

    addSourcesToMessage(messageElement, sources) {
        const sourcesContainer = document.createElement('div');
        sourcesContainer.className = 'sources-container';
        
        const sourcesHeader = document.createElement('div');
        sourcesHeader.className = 'sources-header';
        sourcesHeader.innerHTML = `📚 <strong>Sources:</strong> Found ${sources.length} relevant document${sources.length > 1 ? 's' : ''}`;
        
        const sourcesList = document.createElement('div');
        sourcesList.className = 'sources-list';
        
        sources.forEach((source, index) => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            
            // Extract source information with fallbacks
            const title = source.title || source.metadata?.title || `Document ${index + 1}`;
            const url = source.url || source.metadata?.url || '#';
            const similarity = source.similarity_score || 0;
            const preview = source.chunk_text || source.content_preview || source.content || 'No preview available';
            
            // Create source header with title, link, and similarity score
            const sourceHeader = document.createElement('div');
            sourceHeader.className = 'source-header';
            
            const sourceNumber = document.createElement('span');
            sourceNumber.className = 'source-number';
            sourceNumber.textContent = `${index + 1}`;
            
            const sourceLink = document.createElement('a');
            sourceLink.className = 'source-title-link';
            sourceLink.href = url;
            sourceLink.target = '_blank';
            sourceLink.title = 'Open source document';
            sourceLink.innerHTML = `
                ${this.escapeHtml(title)}
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-left: 4px;">
                    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                    <polyline points="15,3 21,3 21,9"></polyline>
                    <line x1="10" y1="14" x2="21" y2="3"></line>
                </svg>
            `;
            
            const similarityScore = document.createElement('span');
            similarityScore.className = 'similarity-score';
            similarityScore.title = 'Similarity score';
            similarityScore.textContent = `${(similarity * 100).toFixed(1)}%`;
            
            sourceHeader.appendChild(sourceNumber);
            sourceHeader.appendChild(sourceLink);
            sourceHeader.appendChild(similarityScore);
            
            // Create source preview
            const sourcePreview = document.createElement('div');
            sourcePreview.className = 'source-preview';
            sourcePreview.textContent = preview.length > 150 ? preview.substring(0, 150) + '...' : preview;
            
            sourceItem.appendChild(sourceHeader);
            sourceItem.appendChild(sourcePreview);
            sourcesList.appendChild(sourceItem);
        });
        
        sourcesContainer.appendChild(sourcesHeader);
        sourcesContainer.appendChild(sourcesList);
        messageElement.appendChild(sourcesContainer);
    }

    addTypingIndicator() {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';
        
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        
        const typingText = document.createElement('span');
        typingText.textContent = 'Assistant is typing';
        
        const dotsDiv = document.createElement('div');
        dotsDiv.className = 'typing-dots';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'typing-dot';
            dotsDiv.appendChild(dot);
        }
        
        typingDiv.appendChild(typingText);
        typingDiv.appendChild(dotsDiv);
        messageDiv.appendChild(typingDiv);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        return messageDiv;
    }

    addErrorMessage(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = `❌ Error: ${message}`;
        
        this.chatMessages.appendChild(errorDiv);
        this.scrollToBottom();
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    getSimilarityThreshold() {
        const thresholdInput = document.getElementById('similarity-threshold');
        return thresholdInput ? parseFloat(thresholdInput.value) : 0.3;
    }

    addContextInfo(info) {
        const contextDiv = document.createElement('div');
        contextDiv.className = 'context-info';
        contextDiv.innerHTML = `<small><i class="fas fa-info-circle"></i> ${info}</small>`;
        this.chatMessages.appendChild(contextDiv);
        this.scrollToBottom();
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize chat interface when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ChatInterface();
});

// ----------------- Markdown & content utilities -----------------

function stripThinkBlocks(text) {
    return text.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
}

// New utility: split content into main and think parts
function splitThinkBlocks(text) {
    const thinkMatches = Array.from(text.matchAll(/<think>([\s\S]*?)<\/think>/gi));
    const thinkParts = thinkMatches.map(m => m[1].trim()).filter(Boolean);
    const main = text.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
    return { main, think: thinkParts.join('\n\n') };
}

function formatMessageContent(content) {
    try {
        if (typeof marked !== 'undefined') {
            marked.setOptions({
                breaks: false,
                gfm: true,
                sanitize: false,
                pedantic: false,
                smartLists: true
            });
            return marked.parse(content);
        }
    } catch (err) {
        console.error('Markdown parsing error:', err);
    }
    // Fallback – basic HTML escape
    const div = document.createElement('div');
    div.textContent = content;
    return div.innerHTML.replace(/\n/g, '<br>');
}
