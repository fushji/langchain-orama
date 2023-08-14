# Integrate 🦜🔗LangChain with Orama Search

## Introduction
During the last months, for some work projects, I played a lot with [LangChain](https://python.langchain.com/) framework to understand how it can be useful in developing A.I. powered applications.
For those who never hear about it, LangChain is a framework (with a Python and typescript variant) that allow building application that leverages LLM models, in particular, it's suitable for implementing a chatbot, autonomous agent, analysing structured/unstructured data and knowledge management exploration.
In particular, I go deep with this last use case trying to understand how with LangChain, we could explore a knowledge base made up of different documents.
In the realm of information retrieval and semantic search, embeddings and vector databases have emerged as pivotal components underpinning modern search systems' effectiveness and efficiency. 
Embeddings, rooted in natural language processing and machine learning, encode complex data such as text, images, or audio into high-dimensional vector representations that capture semantic relationships and contextual nuances. Leveraging techniques like word2vec, GloVe, or transformer-based models, embeddings enable the translation of intricate data into a mathematical space where semantic similarity can be quantified through vector operations. This leads us to vector databases, specialized data structures designed to store and retrieve these embeddings efficiently. Unlike traditional databases, which index data based on exact matches or predefined criteria, vector databases exploit the geometric properties of embeddings to facilitate semantic search. By calculating distances or similarities between vectors, these databases empower systems to identify and retrieve information that bears a meaningful contextual resemblance, transcending syntactic limitations. The synergy of embeddings and vector databases heralds a new era of semantic search, where users can uncover intricate relationships and patterns within vast datasets with remarkable accuracy and speed, making them indispensable tools across domains like information retrieval, recommendation systems, and content organization.
Langchain comes up with the possibility to use different [Text embedding models](https://js.langchain.com/docs/modules/data_connection/text_embedding/) and different [Vector Store](https://js.langchain.com/docs/modules/data_connection/vectorstores/integrations/) but also allows to extends to other models and vectorstore not yet supported.
So starting from this point I write an integration module for the js implementation of Langchain that using **Orama** as vectorstore for embeddings.

### Orama
[Orama](https://oramasearch.com/) is a "fast, batteries-included, full-text and vector search engine entirely written in TypeScript, with zero dependencies." written by Michele Riva, CTO of the startup OramaSearch.
Orama from version 1.2.0 supports [**vector search**](https://docs.oramasearch.com/usage/create#vector-properties) so I decided to try it integrating with Langchain as `vectorstore`.

## Explore the code
From the LangChain [docs](https://js.langchain.com/docs/modules/data_connection/vectorstores/) a vectorstore must implement the `interface VectorStore` interface.

The `addDocuments` create an instance of Orama db and starting from an array of Document containing the contents read from different sources and for each content generate the related embedding vector using the model passed inside the class constructor, then the method addVector is called.

```typescript
async addDocuments(
    documents: Document<Record<string, any>>[],
    options?: { ids?: string[] }
  ): Promise<void | string[]> {
    this.db = await create({
      schema: {
        id: 'string',
        content: 'string',
        embeddings: 'vector[1536]',
        metadata: 'string'
      },
      id: this.dbName
    });

    if (this.db == undefined) {
      throw new Error(`Valid Orama db instance is required!`);
    }

    const contents = documents.map(({ pageContent }) => pageContent);
    return this.addVectors(
      await this.embeddings.embedDocuments(contents),
      documents,
      options
    );
  }
```
The methods addVector takes the embedding vector, the list of documents with its metadata and call the vectorstore API for content insertions. In this case we can leverage on the batch API `insertMultiple` that Orama provides us for bulk load.

```ts
async addVectors(
    vectors: number[][],
    documents: Document<Record<string, any>>[],
    options?: { ids?: string[] }
  ) {
    if (this.db == undefined) {
      throw new Error(`Valid Orama db instance is required!`);
    }

    if (vectors.length == 0) return [];

    if (vectors.length !== documents.length) {
      throw new Error(`Vectors and documents must have the same length`);
    }

    const documentsId =
      options?.ids ??
      Array.from({ length: documents.length }, () => webcrypto.randomUUID());

    const docsMetadata = documents.map(({ metadata }) => {
      return JSON.stringify(metadata);
    });

    const batchNumber = Math.floor(documents.length / this.BATCH_SIZE);

    for (let r = 0; r < batchNumber; r++) {
      const docs: OramaDocument[] = [];
      for (let i = 0; i < this.BATCH_SIZE; i++) {
        const doc = {
          id: webcrypto.randomUUID(),
          content: documents[r * this.BATCH_SIZE + i].pageContent,
          embeddings: vectors[r * this.BATCH_SIZE + i],
          metadata: docsMetadata[r * this.BATCH_SIZE + i]
        };
        docs.push(doc);
      }

      await insertMultiple(this.db, docs, this.BATCH_SIZE);
    }
    const docs = [];
    for (let i = batchNumber * this.BATCH_SIZE; i < documents.length; i++) {
      const doc = {
        id: webcrypto.randomUUID(),
        content: documents[i].pageContent,
        embeddings: vectors[i],
        metadata: docsMetadata[i]
      };
      docs.push(doc);
    }

    await insertMultiple(this.db, docs, this.BATCH_SIZE);

    return documentsId;
  }
```

`similaritySearchVectorWithScore` methods implements the similarity search using a embedding of the query string and performing a search using the `searchVector` API.

```ts
async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    filter?: this['FilterType'] | undefined
  ): Promise<[Document<Record<string, any>>, number][]> {
    if (this.db == undefined) {
      throw new Error(`Valid Orama db instance is required!`);
    }

    const searchResults = await searchVector(this.db, {
      vector: query, // OpenAI embedding or similar vector to be used as an input
      property: 'embeddings' // Property to search through. Mandatory for vector search
    });

    if (this.db == undefined) {
      throw new Error(`Valid Orama db instance is required!`);
    }
    if (searchResults.count == 0) {
      return [];
    }
    const results: [Document, number][] = [];
    for (let i = 0; i < k && i < searchResults.count; i++) {
      const doc = searchResults.hits[i];

      const metadata: Document['metadata'] = JSON.parse(
        doc.document.metadata as string
      );

      results.push([
        new Document({
          pageContent: (doc.document?.content as string) ?? '',
          metadata
        }),
        doc.score
      ]);
    }

    return results;
  }
```

The following methods are two high-level API that we can use in our code to load in our database raw text string or documents.

```ts
static async fromTexts(
    texts: string[],
    metadatas: object[] | object,
    embeddings: Embeddings,
    args: OramaArgs
  ): Promise<OramaSearch> {
    const docs: Document[] = [];
    for (let i = 0; i < texts.length; i += 1) {
      const metadata = Array.isArray(metadatas) ? metadatas[i] : metadatas;
      const newDoc = new Document({
        pageContent: texts[i],
        metadata
      });
      docs.push(newDoc);
    }
    return this.fromDocuments(docs, embeddings, args);
  }

  static async fromDocuments(
    docs: Document[],
    embeddings: Embeddings,
    args: OramaArgs
  ): Promise<OramaSearch> {
    const instance = new this(embeddings, args);
    await instance.addDocuments(docs);
    return instance;
  }
  ```

## Let's see in action

After seeing the internals of our `OramaSearch` class let's see how it works in a complete workflow. The following example is an adaptation of code you can find [here](https://js.langchain.com/docs/modules/data_connection/vectorstores/integrations/).

```ts
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
```
The complete code is available on [GitHub](https://github.com/fushji/langchain-orama).

## Sum up
In this article, we explore how to use Orama search engine as vectorstore backend for Langchain in implementing a semantic search application.
To complete the workflow after Load, Transform, Embed, and Store need a [Retrieve](https://js.langchain.com/docs/modules/data_connection/retrievers/) step using the public API of the `BaseRetriever` class.

