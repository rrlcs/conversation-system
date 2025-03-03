import React, { useState } from 'react';
import {
  Box,
  Container,
  TextField,
  Button,
  Paper,
  Typography,
  CircularProgress,
  Chip,
  IconButton,
  Collapse
} from '@mui/material';
import { Send as SendIcon, ExpandMore, ExpandLess } from '@mui/icons-material';
import axios from 'axios';

function App() {
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [expandedMessage, setExpandedMessage] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    const newMessage = { question, timestamp: new Date().toISOString() };
    setMessages(prev => [...prev, newMessage]);

    try {
      const response = await axios.post('http://localhost:8000/question', {
        question: question.trim()
      });

      setMessages(prev => [
        ...prev.slice(0, -1),
        {
          ...newMessage,
          answer: response.data.answer,
          classification: response.data.classification,
          verification: response.data.verification_result,
          context: response.data.context
        }
      ]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [
        ...prev.slice(0, -1),
        {
          ...newMessage,
          answer: 'Sorry, there was an error processing your question.',
          error: true
        }
      ]);
    }

    setLoading(false);
    setQuestion('');
  };

  const toggleExpand = (index) => {
    setExpandedMessage(expandedMessage === index ? null : index);
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Conversation System
        </Typography>
        
        <Paper 
          sx={{ 
            height: '60vh', 
            overflowY: 'auto', 
            p: 2, 
            mb: 2,
            backgroundColor: '#f5f5f5'
          }}
        >
          {messages.map((message, index) => (
            <Box key={index} sx={{ mb: 2 }}>
              {/* User Question */}
              <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 1 }}>
                <Paper 
                  sx={{ 
                    p: 2, 
                    backgroundColor: '#e3f2fd',
                    maxWidth: '80%'
                  }}
                >
                  <Typography>{message.question}</Typography>
                </Paper>
              </Box>

              {/* AI Response */}
              {message.answer && (
                <Box sx={{ display: 'flex', justifyContent: 'flex-start', mb: 1 }}>
                  <Paper 
                    sx={{ 
                      p: 2,
                      backgroundColor: '#fff',
                      maxWidth: '80%'
                    }}
                  >
                    <Typography>{message.answer}</Typography>
                    
                    {/* Classification and Verification */}
                    {message.classification && (
                      <Box sx={{ mt: 1, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                        <Chip 
                          label={message.classification}
                          color="primary"
                          size="small"
                        />
                        <Chip 
                          label={message.verification ? "Verified" : "Unverified"}
                          color={message.verification ? "success" : "warning"}
                          size="small"
                        />
                      </Box>
                    )}

                    {/* Context Information */}
                    {message.context && (
                      <>
                        <IconButton 
                          size="small" 
                          onClick={() => toggleExpand(index)}
                          sx={{ mt: 1 }}
                        >
                          {expandedMessage === index ? <ExpandLess /> : <ExpandMore />}
                        </IconButton>
                        <Collapse in={expandedMessage === index}>
                          <Box sx={{ mt: 1, p: 1, backgroundColor: '#f5f5f5', borderRadius: 1 }}>
                            <Typography variant="caption" component="div">
                              <strong>Context:</strong>
                            </Typography>
                            {message.context.retrieval_results.map((result, i) => (
                              <Typography key={i} variant="caption" component="div">
                                {result.text}
                              </Typography>
                            ))}
                          </Box>
                        </Collapse>
                      </>
                    )}
                  </Paper>
                </Box>
              )}
            </Box>
          ))}
          {loading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
              <CircularProgress />
            </Box>
          )}
        </Paper>

        {/* Question Input */}
        <form onSubmit={handleSubmit}>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Ask a question..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={loading}
            />
            <Button
              variant="contained"
              color="primary"
              type="submit"
              disabled={loading || !question.trim()}
              sx={{ minWidth: '100px' }}
            >
              {loading ? <CircularProgress size={24} /> : <SendIcon />}
            </Button>
          </Box>
        </form>
      </Box>
    </Container>
  );
}

export default App; 