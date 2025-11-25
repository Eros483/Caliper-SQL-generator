const API_URL = "http://localhost:8000"; // Or whatever your URL is

export const checkHealth = async () => {
  const response = await fetch(`${API_URL}/health`);
  return response.json();
};

// UPDATE THIS FUNCTION
export const sendMessage = async (query, sessionId) => {
  const response = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ 
      query: query,
      session_id: sessionId // <--- Add this field
    }),
  });

  if (!response.ok) {
    throw new Error('Failed to send message');
  }

  return response.json();
};