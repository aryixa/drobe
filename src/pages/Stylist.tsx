import React, { useState } from 'react'
import { MessageSquare, Send, Bot, User } from 'lucide-react'
import { Button } from '../components/ui/Button'

interface Message {
  id: string
  text: string
  sender: 'user' | 'bot'
  timestamp: Date
}

export function Stylist() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: "Hello! I'm your personal stylist assistant. I can help you create outfits, suggest combinations, and provide styling tips. How can I help you today?",
      sender: 'bot',
      timestamp: new Date()
    }
  ])
  const [inputText, setInputText] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const handleSendMessage = async () => {
    if (!inputText.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputText.trim(),
      sender: 'user',
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputText('')
    setIsLoading(true)

    // Simulate bot response
    setTimeout(() => {
      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: getBotResponse(inputText),
        sender: 'bot',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, botResponse])
      setIsLoading(false)
    }, 1000)
  }

  const getBotResponse = (userInput: string): string => {
    const input = userInput.toLowerCase()
    
    if (input.includes('outfit') || input.includes('wear')) {
      return "I'd be happy to help you create an outfit! Based on your wardrobe, I recommend starting with a versatile top and building around it. Try pairing a neutral-colored top with complementary bottoms. Don't forget accessories to complete the look!"
    }
    
    if (input.includes('color') || input.includes('match')) {
      return "Color coordination is key! For a classic look, try complementary colors like blue and orange, or analogous colors like different shades of blue. Neutral colors (black, white, gray, navy) are versatile and work with almost everything."
    }
    
    if (input.includes('season') || input.includes('weather')) {
      return "Seasonal dressing is important! In summer, opt for breathable fabrics like cotton and linen. For winter, layering is your best friend - start with a base layer and add pieces you can remove as needed. Don't forget to check the weather forecast!"
    }
    
    if (input.includes('occasion') || input.includes('event')) {
      return "Consider the dress code and formality of your occasion. For casual events, comfort is key. For business meetings, opt for professional separates. For formal events, choose elegant pieces that make you feel confident."
    }
    
    return "That's a great question! I recommend thinking about the occasion, weather, and your personal style. Start with a foundation piece you love and build around it. Remember, confidence is the best accessory!"
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-primary-900 mb-2">AI Stylist</h1>
        <p className="text-primary-600">
          Get personalized styling advice and outfit recommendations powered by AI.
        </p>
      </div>

      {/* Chat Container */}
      <div className="card h-[600px] flex flex-col">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex items-start space-x-3 ${
                message.sender === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {message.sender === 'bot' && (
                <div className="flex-shrink-0 w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
                  <Bot className="w-4 h-4 text-primary-600" />
                </div>
              )}
              
              <div
                className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                  message.sender === 'user'
                    ? 'bg-primary-900 text-white'
                    : 'bg-primary-100 text-primary-900'
                }`}
              >
                <p className="text-sm">{message.text}</p>
                <p className={`text-xs mt-1 ${
                  message.sender === 'user' ? 'text-primary-200' : 'text-primary-500'
                }`}>
                  {message.timestamp.toLocaleTimeString([], { 
                    hour: '2-digit', 
                    minute: '2-digit' 
                  })}
                </p>
              </div>

              {message.sender === 'user' && (
                <div className="flex-shrink-0 w-8 h-8 bg-primary-900 rounded-full flex items-center justify-center">
                  <User className="w-4 h-4 text-white" />
                </div>
              )}
            </div>
          ))}
          
          {isLoading && (
            <div className="flex items-start space-x-3 justify-start">
              <div className="flex-shrink-0 w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
                <Bot className="w-4 h-4 text-primary-600" />
              </div>
              <div className="bg-primary-100 text-primary-900 px-4 py-2 rounded-lg">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-primary-400 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-primary-400 rounded-full animate-bounce delay-100" />
                  <div className="w-2 h-2 bg-primary-400 rounded-full animate-bounce delay-200" />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Input */}
        <div className="border-t border-primary-200 p-4">
          <div className="flex space-x-3">
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask for styling advice..."
              className="flex-1 px-3 py-2 border border-primary-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
              disabled={isLoading}
            />
            <Button
              onClick={handleSendMessage}
              disabled={!inputText.trim() || isLoading}
              size="sm"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
          
          {/* Quick Suggestions */}
          <div className="mt-3 flex flex-wrap gap-2">
            {[
              "What should I wear today?",
              "Help me create an outfit",
              "Color matching tips",
              "Seasonal styling"
            ].map((suggestion) => (
              <button
                key={suggestion}
                onClick={() => setInputText(suggestion)}
                className="px-3 py-1 text-xs bg-primary-100 text-primary-700 rounded-full hover:bg-primary-200 transition-colors"
                disabled={isLoading}
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Tips Section */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card p-6 text-center">
          <div className="mx-auto w-12 h-12 bg-accent-100 rounded-full flex items-center justify-center mb-4">
            <MessageSquare className="w-6 h-6 text-accent-600" />
          </div>
          <h3 className="font-semibold text-primary-900 mb-2">Personalized Advice</h3>
          <p className="text-sm text-primary-600">
            Get styling tips tailored to your wardrobe and preferences
          </p>
        </div>
        
        <div className="card p-6 text-center">
          <div className="mx-auto w-12 h-12 bg-accent-100 rounded-full flex items-center justify-center mb-4">
            <Bot className="w-6 h-6 text-accent-600" />
          </div>
          <h3 className="font-semibold text-primary-900 mb-2">AI Powered</h3>
          <p className="text-sm text-primary-600">
            Smart recommendations based on fashion best practices
          </p>
        </div>
        
        <div className="card p-6 text-center">
          <div className="mx-auto w-12 h-12 bg-accent-100 rounded-full flex items-center justify-center mb-4">
            <Send className="w-6 h-6 text-accent-600" />
          </div>
          <h3 className="font-semibold text-primary-900 mb-2">Instant Help</h3>
          <p className="text-sm text-primary-600">
            Quick answers to all your styling questions
          </p>
        </div>
      </div>
    </div>
  )
}
