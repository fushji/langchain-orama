import 'dotenv/config';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OramaSearch } from './orama.js';

import { CheerioWebBaseLoader } from 'langchain/document_loaders/web/cheerio';

const loader = new CheerioWebBaseLoader(
  'https://lilianweng.github.io/posts/2023-06-23-agent/'
);
const data = await loader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 0
});

const splitDocs = await textSplitter.splitDocuments(data);

const vectorStore = await OramaSearch.fromDocuments(
  splitDocs,
  new OpenAIEmbeddings(),
  {
    dbName: 'orama-test'
  }
);

const relevantDocs = await vectorStore.similaritySearch(
  'What is task decomposition?'
);

console.log(relevantDocs.length);
console.log(relevantDocs);
