import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { PromptTemplate } from 'langchain/prompts';
import { BufferMemory } from "langchain/memory";

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `You are an AI assistant for Notion. You are given the following extracted parts of a long document and a question. Provide a conversational answer based on the context provided.
You should only use hyperlinks as references that are explicitly listed as a source in the context below. Do NOT make up a hyperlink that is not listed below.
If you can't find the answer in the context below, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not related to Notion, notion api or the context provided, politely inform them that you are tuned to only answer questions that are related to Notion.
Choose the most relevant link that matches the context provided:

Question: {question}
=========
Context: {context}
=========
Chat History: {chat_history}
=========
Answer in Markdown:`,
);

export const makeChain = (vectorstore: SupabaseVectorStore) => {
  const model = new ChatOpenAI({
    temperature: 0,
    streaming: true,
    modelName: "gpt-4-1106-preview"
});

return ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever({
        verbose: true
    }),
    {
        verbose: true,
        returnSourceDocuments: true,
        qaChainOptions: {
            type: "stuff",
            prompt: QA_PROMPT
        },
        memory: new BufferMemory({
            memoryKey: "chat_history", // Must be set to "chat_history",
            inputKey: "question",
            outputKey: "text"
        }),
        questionGeneratorChainOptions: {
            llm: new ChatOpenAI({ temperature: 0 }),
            template: CONDENSE_PROMPT.template
        }
    }
)};
