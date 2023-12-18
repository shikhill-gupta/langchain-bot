import { BaseCallbackHandler, CallbackManager } from 'langchain/callbacks';
import { OpenAI } from 'langchain/llms/openai';

if (!process.env.OPENAI_API_KEY) {
  throw new Error('Missing OpenAI Credentials');
}

export const openai = new OpenAI({
  temperature: 0,
});

const callbackManager = new CallbackManager();
callbackManager.addHandler(BaseCallbackHandler.fromMethods({

  handleLLMNewToken(token) {
    console.log(token);
  }
}));

export const openaiStream = new OpenAI({
  temperature: 0,
  streaming: true,
  callbacks: callbackManager
});
