import type { NextApiRequest, NextApiResponse } from 'next';
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { SupabaseLibArgs, SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { supabaseClient } from '@/utils/supabase-client';
import { makeChain } from '@/utils/makechain';
import { ConversationalRetrievalQAChain } from "langchain/chains";

let chain: ConversationalRetrievalQAChain | null = null;
export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  const { question, history } = req.body;

  if (!question) {
    return res.status(400).json({ message: 'No question in the request' });
  }
  // OpenAI recommends replacing newlines with spaces for best results
  const sanitizedQuestion = question.trim().replaceAll('\n', ' ');

  const dbConfig: SupabaseLibArgs = {
    client: supabaseClient,
    tableName: "documents",
    // ...other Supabase client config 
  };
  /* create vectorstore*/
  const vectorStore = await SupabaseVectorStore.fromExistingIndex(
    new OpenAIEmbeddings(),
    dbConfig
  );

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
  });

  const sendData = (data: string) => {
    res.write(`data: ${data}\n\n`);
  };

  sendData(JSON.stringify({ data: '' }));

  // create the chain

  if (!chain) {
    chain = makeChain(vectorStore);
  }

  try {
    //Ask a question
    const response = await chain.call({ question: sanitizedQuestion }, [
      {
        handleLLMNewToken(token: string) {
          sendData(JSON.stringify({ data: token }));
        },
      },
    ]);

    console.log('response', response);
  } catch (error) {
    console.log('error', error);
  } finally {
    sendData('[DONE]');
    res.end();
  }
}
