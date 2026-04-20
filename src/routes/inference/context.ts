import { createContext } from "svelte";

export type InferenceContext = {
	tokens: string[];
	attentionWeights: number[][];
};

export const [getInferenceContext, setInferenceContext] = createContext<InferenceContext>();
