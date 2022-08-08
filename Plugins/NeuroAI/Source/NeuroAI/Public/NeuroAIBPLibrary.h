// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "NeuroAITypes.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "NeuroAIBPLibrary.generated.h"

/* 
*	Neuro Function library class.
*
*/
UCLASS()
class UNeuroAIBPLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_UCLASS_BODY()

	UFUNCTION(BlueprintCallable, meta = (DisplayName = "Neuro Activation Function", Keywords = "Returns result of specified activation function on given input."), Category = "NeuroAI")
	static TArray<float> NeuroActivationFunction(TArray<float> Input, ENeuroActivationFunction ActivationType, bool bClamped = true);

	//The actual activation functions, used by the above internally
	static TArray<float> NeuroActivationFunction_LinearRectified(TArray<float> Input);
	static TArray<float> NeuroActivationFunction_LeakyLinearRectified(TArray<float> Input);
	static TArray<float> NeuroActivationFunction_BinaryStep(TArray<float> Input);
	static TArray<float> NeuroActivationFunction_Sigmoid(TArray<float> Input);
	static TArray<float> NeuroActivationFunction_TanH(TArray<float> Input);
	static TArray<float> NeuroActivationFunction_Swish(TArray<float> Input);
	static TArray<float> NeuroActivationFunction_ArgMax(TArray<float> Input);
	static TArray<float> NeuroActivationFunction_SoftMax(TArray<float> Input);
	static TArray<float> NeuroActivationFunction_Exponential(TArray<float> Input);

	// Generate a randomized neural network within given constraints
	// ReLU activation function should only be used in the hidden layers.
	// Sigmoid/Logistic and Tanh functions should not be used in hidden layers as they make the model more susceptible to problems during training (due to vanishing gradients).
	// Swish function is used in neural networks having a depth greater than 40 layers.
	UFUNCTION(BlueprintCallable, Category = "NeuroAI")
	static FNeuroLobe GenerateRandomNeuroLobe(TArray<FName> InputNames, TArray<FName> OutputNames, int32 NumHiddenLayers, int32 HiddenLayerSize, ENeuroActivationFunction InputFunction = NAct_None,
	ENeuroActivationFunction HLFunction = NAct_RectLinear, ENeuroActivationFunction OutputFunction = NAct_Sigmoid);

	// Generates a layer of nodes with a given activation function
	static FNeuroLayer GenerateRandomNeuroLayer(int32 NumNodes, ENeuroActivationFunction ActivationFunction, int32 NumPreviousLayerNodes);

	// Mutates the weights and biases by a given threshold to create a new lobe from an existing one
	UFUNCTION(BlueprintCallable, Category = "NeuroAI")
	static FNeuroLobe MutateLobeSimple(const FNeuroLobe InLobe, int32 NumWeightMutations, int32 NumBiasesMutations, float MaximumDeltaWeights, float
	                                   MaximumDeltaBiases);

	// Mutates the lobe by inserting a new layer, while preserving existing behavior
	UFUNCTION(BlueprintCallable, Category = "NeuroAI")
	static FNeuroLobe MutateLobeNewHiddenLayer(const FNeuroLobe InLobe);
	
	// Returns true if the lobes are equivalent in structure and thus compatible for simple breeding
	UFUNCTION(BlueprintCallable, Category = "NeuroAI")
	static bool AreLobesHomologous(const FNeuroLobe A, const FNeuroLobe B);

	// Create a new lobe via a combination of the weights and biases of two existing compatible lobes
	UFUNCTION(BlueprintCallable, Category = "NeuroAI")
	static FNeuroLobe BreedHomologousLobesSimple(const FNeuroLobe A, const FNeuroLobe B);

	// Crete a new generation of lobes by breeding the first pair
	UFUNCTION(BlueprintCallable, Category = "NeuroAI")
	static FNeuroGeneration BreedNewGeneration(const FNeuroGeneration InGeneration, int32 NumBreeding, int32 OffspringPerPair);

	// Create a new generation by adding new inputs to an existing generation of lobes
	UFUNCTION(BlueprintCallable, Category = "NeuroAI")
	static FNeuroGeneration MutateGenerationAddInputs(const FNeuroGeneration InGeneration, TArray<FName> InputNames);

	UFUNCTION(BlueprintCallable, Category = "NeuroAI")
	static FNeuroGeneration MutateGenerationAddLayer(const FNeuroGeneration InGeneration);

	// Create a new generation by removing inputs from an existing generation
	UFUNCTION(BlueprintCallable, Category = "NeuroAI")
	static FNeuroGeneration MutateGenerationRemoveInputs(const FNeuroGeneration InGeneration, TArray<int32> InputIndicesToRemove);

	UFUNCTION(BlueprintCallable, Category = "NeuroAI")
	static FNeuroGeneration MutateGenerationSimple(const FNeuroGeneration InGeneration, int32 NumWeightMutations, int32 NumBiasesMutations, float MaximumDeltaWeights, float MaximumDeltaBiases);

	// Append the highest scoring lobes from one generation into another generation
	UFUNCTION(BlueprintCallable, Category = "NeuroAI")
	static FNeuroGeneration SurviveLobes(const FNeuroGeneration SurviveFrom, const FNeuroGeneration SurviveTo, int32 NumToSurvive);

	UFUNCTION(BlueprintCallable, Category = "NeuroAI")
	static FNeuroLineage AppendGenerationToLineage(const FNeuroLineage InLineage, const FNeuroGeneration InGeneration);

	UFUNCTION(BlueprintCallable, Category = "NeuroAI")
	static FNeuroGeneration SetGenerationScores(const FNeuroGeneration InGeneration, const TArray<float> InScores);

	UFUNCTION(BlueprintCallable, Category = "NeuroAI")
	static FNeuroLineage SetLastGenerationScores(const FNeuroLineage InLineage, const TArray<float> InScores);

	UFUNCTION(BlueprintCallable, Category = "NeuroAI")
	static FNeuroGeneration SetGenerationLobes(const FNeuroGeneration InGeneration, const TArray<FNeuroLobe> InLobes);

	UFUNCTION(BlueprintCallable, Category = "NeuroAI")
	static FNeuroGeneration GetLatestGeneration(const FNeuroLineage InLineage);

	UFUNCTION(BlueprintCallable, Category = "NeuroAI")
	static TArray<FNeuroLobe> GetGenerationLobes(const FNeuroGeneration InGeneration);

	UFUNCTION(BlueprintCallable, Category = "NeuroAI")
	static TArray<float> EvaluateLobe(FNeuroLobe InLobe, const TArray<float> Inputs);
};
