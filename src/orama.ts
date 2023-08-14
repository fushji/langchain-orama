import { webcrypto } from 'node:crypto';
import { Document } from 'langchain/document';
import { Embeddings } from 'langchain/embeddings';
import { create, insertMultiple, searchVector } from '@orama/orama';
import type { Orama, Document as OramaDocument } from '@orama/orama';
import { VectorStore } from 'langchain/vectorstores/base';
export type OramaArgs = { dbName?: string };

export class OramaSearch extends VectorStore {
  readonly BATCH_SIZE = 500;
  dbName?: string;
  db?: Orama | null;

  constructor(embeddings: Embeddings, args: OramaArgs) {
    super(embeddings, args);
    this.dbName = setDbName(args.dbName);
  }

  _vectorstoreType(): string {
    return 'orama';
  }

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
}

function setDbName(dbName?: string) {
  if (!dbName) {
    return `orama-${webcrypto.randomUUID()}`;
  }
  return dbName;
}
