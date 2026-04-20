import { createContext } from "svelte";

// [layer][head][token][token]
export type AttentionWeights = number[][][][];

export type InferenceContext = {
	tokens: string[];
	attentionWeights: AttentionWeights;
	selectedLayer: number | null;
};

export const [getInferenceContext, setInferenceContext] = createContext<InferenceContext>();
