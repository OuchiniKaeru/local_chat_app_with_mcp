document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const chatWindow = document.getElementById('chat-window');
    const chatHistoryList = document.getElementById('chat-history-list');
    const newChatBtn = document.getElementById('new-chat-btn');
    const fileUploadBtn = document.getElementById('file-upload-btn');
    const fileInput = document.getElementById('file-input');
    const settingsBtn = document.getElementById('settings-btn');
    const settingsModal = document.getElementById('settings-modal');
    const mcpBtn = document.getElementById('mcp-btn');
    const mcpModal = document.getElementById('mcp-modal');
    const editSettingsJsonBtn = document.getElementById('edit-settings-json-btn');
    const editRawSettingsModal = document.getElementById('edit-raw-settings-modal');

    let currentChatId = null;

    // --- Display Functions ---
    const displayMessage = (sender, message, thoughtProcess = []) => {
        const messageContainer = document.createElement('div');
        messageContainer.classList.add('message-container', sender === 'user' ? 'user-message' : 'ai-message');

        const messageBubble = document.createElement('div');
        messageBubble.classList.add('message-bubble');
        messageBubble.textContent = message;
        messageContainer.appendChild(messageBubble);

        if (sender === 'ai' && thoughtProcess.length > 0) {
            addThoughtProcessElements(messageContainer, thoughtProcess);
        }

        chatWindow.appendChild(messageContainer);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    };

    const addThoughtProcessElements = (messageContainer, thoughts) => {
        const details = document.createElement('details');
        details.classList.add('thought-process');
        const summary = document.createElement('summary');
        summary.textContent = 'Show Thoughts';
        details.appendChild(summary);

        const content = document.createElement('div');
        content.innerHTML = thoughts.map(t => `<p>${t.replace(/\n/g, '<br>')}</p>`).join('');
        details.appendChild(content);

        // Insert before the message bubble
        messageContainer.insertBefore(details, messageContainer.firstChild);
    };

    // --- API & Chat Logic ---
    const fetchChatHistory = async () => {
        try {
            const response = await fetch('/api/chats');
            const chats = await response.json();
            console.log('Fetched chat history:', chats); // Debug log
            chatHistoryList.innerHTML = '';
            chats.forEach(chat => {
                const li = document.createElement('li');
                li.classList.add('history-item');
                li.textContent = chat.title;
                li.dataset.chatId = chat.id;
                li.addEventListener('click', () => loadChat(chat.id));
                chatHistoryList.appendChild(li);
            });
        } catch (error) {
            console.error('Error fetching chat history:', error);
        }
    };

    const loadChat = async (chatId) => {
        try {
            const response = await fetch(`/api/chat/${chatId}`);
            const messages = await response.json();
            chatWindow.innerHTML = '';
            currentChatId = chatId;

            document.querySelectorAll('.history-item').forEach(item => {
                item.classList.toggle('active', item.dataset.chatId === chatId);
            });

            messages.forEach(msg => {
                displayMessage(msg.sender, msg.message, msg.thought_process);
            });
            chatWindow.dataset.chatId = chatId; // Set data-chat-id
        } catch (error) {
            console.error('Error loading chat:', error);
        }
    };

    const handleFormSubmit = async (e) => {
        e.preventDefault();
        const message = messageInput.value.trim();
        if (!message) return;

        displayMessage('user', message);
        messageInput.value = '';

        const aiMessageContainer = document.createElement('div');
        aiMessageContainer.classList.add('message-container', 'ai-message');
        const aiMessageBubble = document.createElement('div');
        aiMessageBubble.classList.add('message-bubble');
        const cursor = document.createElement('span');
        cursor.classList.add('typing-cursor');
        cursor.textContent = '▋';
        aiMessageBubble.appendChild(cursor);
        aiMessageContainer.appendChild(aiMessageBubble);
        chatWindow.appendChild(aiMessageContainer);
        chatWindow.scrollTop = chatWindow.scrollHeight;

        let thoughtProcessContainer = null; // To hold thought process details
        let thoughts = [];

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message, chat_id: currentChatId }),
            });

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let receivedData = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                receivedData += decoder.decode(value, { stream: true });
                const lines = receivedData.split('\n\n');
                receivedData = lines.pop(); // Keep incomplete data for next chunk

                for (const line of lines) {
                    if (!line.startsWith('data:')) continue;
                    const jsonString = line.substring(5).trim();
                    try {
                        const data = JSON.parse(jsonString);

                        if (data.type === 'start' && !currentChatId) {
                            currentChatId = data.chat_id;
                            chatWindow.dataset.chatId = currentChatId; // Set data-chat-id for new chat
                            console.log('New chat started, fetching history...'); // Debug log
                            fetchChatHistory(); // Refresh list to show new chat
                        } else if (data.type === 'TEXT_MESSAGE') {
                            cursor.remove();
                            aiMessageBubble.textContent += data.content;
                            aiMessageBubble.appendChild(cursor);
                        } else if (data.type === 'TOOL_CALL' || data.type === 'TOOL_RESULT') {
                            if (!thoughtProcessContainer) {
                                const details = document.createElement('details');
                                details.classList.add('thought-process');
                                const summary = document.createElement('summary');
                                summary.textContent = 'Show Thoughts';
                                details.appendChild(summary);
                                thoughtProcessContainer = document.createElement('div');
                                details.appendChild(thoughtProcessContainer);
                                aiMessageContainer.insertBefore(details, aiMessageBubble);
                            }
                            const p = document.createElement('p');
                            p.innerHTML = `<b>${data.type}:</b> ${data.content.replace(/\n/g, '<br>')}`;
                            thoughtProcessContainer.appendChild(p);
                            thoughts.push(data.content);
                        } else if (data.type === 'end') {
                            cursor.remove();
                            aiMessageBubble.textContent = data.final_reply || aiMessageBubble.textContent;
                            if (thoughtProcessContainer && data.thought_process) {
                                // Final update to thoughts if needed, though usually captured already
                            }
                        } else if (data.type === 'error') {
                            aiMessageBubble.textContent = `Error: ${data.content}`;
                            aiMessageBubble.classList.add('error');
                            cursor.remove();
                        }

                    } catch (parseError) {
                        console.error('Error parsing JSON from stream:', parseError, jsonString);
                    }
                }
            }
        } catch (error) {
            console.error('Error sending message:', error);
            aiMessageBubble.textContent = 'Sorry, something went wrong.';
            cursor.remove();
        }
    };

    // --- Modal & Settings Logic ---
    const setupModal = (buttonId, modalId, onOpenCallback = null) => {
        const btn = document.getElementById(buttonId);
        const modal = document.getElementById(modalId);
        const closeBtn = modal.querySelector('.close-button');

        btn.addEventListener('click', () => {
            modal.style.display = 'block';
            if (onOpenCallback) onOpenCallback();
        });
        closeBtn.addEventListener('click', () => { modal.style.display = 'none'; });
        window.addEventListener('click', (event) => {
            if (event.target === modal) modal.style.display = 'none';
        });
    };

    const loadSettingsIntoModal = async () => {
        try {
            const response = await fetch('/api/settings');
            const settings = await response.json();
            document.getElementById('user-id').value = settings.user_id || '';
            document.getElementById('password').value = settings.password || '';
            document.getElementById('theme').value = settings.theme || 'dark';
            document.getElementById('system-prompt').value = settings.system_prompt || '';
        } catch (error) {
            console.error('Error loading settings:', error);
            alert('Failed to load settings.');
        }
    };

    const loadRawSettingsIntoModal = async () => {
        try {
            const response = await fetch('/api/settings');
            const settings = await response.json();
            document.getElementById('raw-settings-json').value = JSON.stringify(settings, null, 4);
        } catch (error) {
            console.error('Error loading raw settings:', error);
            alert('Failed to load raw settings.');
        }
    };

    const populateMcpServers = async () => {
        const mcpListDiv = document.getElementById('mcp-list');
        mcpListDiv.innerHTML = '';
        try {
            const response = await fetch('/api/settings');
            const settings = await response.json();
            const mcpServers = settings.mcpServers || {};
            if (Object.keys(mcpServers).length === 0) {
                mcpListDiv.innerHTML = '<p>No MCP servers configured.</p>';
                return;
            }
            Object.entries(mcpServers).forEach(([name, config]) => {
                const p = document.createElement('p');
                p.textContent = `${name}: ${config.type} - ${config.command || config.url}`;
                mcpListDiv.appendChild(p);
            });
        } catch (error) {
            console.error('Error fetching MCP servers:', error);
            mcpListDiv.innerHTML = '<p>Error loading MCP servers.</p>';
        }
    };

    // --- Event Listeners ---
    chatForm.addEventListener('submit', handleFormSubmit);
    newChatBtn.addEventListener('click', () => {
        currentChatId = null;
        chatWindow.innerHTML = '';
        chatWindow.dataset.chatId = ''; // Clear data-chat-id
        document.querySelectorAll('.history-item').forEach(item => item.classList.remove('active'));
    });
    fileUploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', async () => { /* ... file upload logic ... */ });

    // Setup modals
    setupModal('settings-btn', 'settings-modal', loadSettingsIntoModal);
    setupModal('mcp-btn', 'mcp-modal', populateMcpServers);
    setupModal('edit-settings-json-btn', 'edit-raw-settings-modal', loadRawSettingsIntoModal);

    // Raw settings editor logic
    document.getElementById('save-raw-settings-btn').addEventListener('click', async () => {
        try {
            const updatedSettings = JSON.parse(document.getElementById('raw-settings-json').value);
            const response = await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updatedSettings),
            });
            if (!response.ok) throw new Error('Failed to save settings');
            alert('Settings saved! Please restart the application for changes to take effect.');
            document.getElementById('edit-raw-settings-modal').style.display = 'none';
        } catch (error) {
            alert('Error saving settings. Please check JSON format.');
            console.error('Error saving raw settings:', error);
        }
    });

    // Initial Load
    fetchChatHistory();
});
