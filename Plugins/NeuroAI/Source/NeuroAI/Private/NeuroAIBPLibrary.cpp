// Copyright Epic Games, Inc. All Rights Reserved.

#include "NeuroAIBPLibrary.h"

#include "Kismet/KismetMathLibrary.h"

UNeuroAIBPLibrary::UNeuroAIBPLibrary(const FObjectInitializer& ObjectInitializer)
: Super(ObjectInitializer)
{

}

TArray<float> UNeuroAIBPLibrary::NeuroActivationFunction(TArray<float> Input, ENeuroActivationFunction ActivationType, bool bClamped)
{
	TArray<float> Output;
	switch(ActivationType)
	{
	case ENeuroActivationFunction::NAct_RectLinear:
		{
			Output = NeuroActivationFunction_LinearRectified(Input);
		}
	case ENeuroActivationFunction::NAct_LeakyRectLinear:
		{
			Output = NeuroActivationFunction_LeakyLinearRectified(Input);
		}
	case ENeuroActivationFunction::NAct_BinaryStep:
		{
			Output = NeuroActivationFunction_BinaryStep(Input);
		}
	case ENeuroActivationFunction::NAct_Sigmoid:
		{
			Output = NeuroActivationFunction_Sigmoid(Input);
		}
	case ENeuroActivationFunction::NAct_TanH:
		{
			Output = NeuroActivationFunction_TanH(Input);
		}
	case ENeuroActivationFunction::NAct_Swish:
		{
			Output = NeuroActivationFunction_Swish(Input);
		}
	default:
		{
			Output = Input;
		}
	}
	//Clamping to a -1 to 1 range is common AI design and helps prevent values from exploding in magnitude as they feed forward
	if(bClamped)
	{
		for(int32 I = 0; I < Output.Num(); I ++)
		{
			Output[I] = FMath::Clamp(Output[I], -1.f, 1.f);
		}
	}

	return Output;
}

TArray<float> UNeuroAIBPLibrary::NeuroActivationFunction_LinearRectified(TArray<float> Input)
{
	TArray<float> Output;

	for(int32 I = 0; I < Input.Num(); I ++)
	{
		Output.Add(FMath::Max(0.f, Input[I]));
	}

	return Output;
}

TArray<float> UNeuroAIBPLibrary::NeuroActivationFunction_LeakyLinearRectified(TArray<float> Input)
{
	TArray<float> Output;

	for(int32 I = 0; I < Input.Num(); I ++)
	{
		Output.Add(FMath::Max(0.1f, Input[I]));
	}

	return Output;
}

TArray<float> UNeuroAIBPLibrary::NeuroActivationFunction_BinaryStep(TArray<float> Input)
{
	TArray<float> Output;

	for(int32 I = 0; I < Input.Num(); I ++)
	{
		Output.Add(Input[I] >= 0.f);
	}

	return Output;
}

TArray<float> UNeuroAIBPLibrary::NeuroActivationFunction_Sigmoid(TArray<float> Input)
{
	TArray<float> Output;

	for(int32 I = 0; I < Input.Num(); I ++)
	{
		Output.Add(1.f / (1 + FMath::Loge(-Input[I])));
	}

	return Output;
}

TArray<float> UNeuroAIBPLibrary::NeuroActivationFunction_TanH(TArray<float> Input)
{
	TArray<float> Output;

	for(int32 I = 0; I < Input.Num(); I ++)
	{
		const float Value = (FMath::Loge(Input[I]) - FMath::Loge(-Input[I])) / (FMath::Loge(Input[I]) + FMath::Loge(-Input[I]));
		Output.Add(Value);
	}

	return Output;
}

TArray<float> UNeuroAIBPLibrary::NeuroActivationFunction_Swish(TArray<float> Input)
{
	TArray<float> Output;
	
	for(int32 I = 0; I < Input.Num(); I ++)
	{
		Output.Add(Input[I] * (Output.Add(1.f / (1 + FMath::Loge(-Input[I])))));
	}
	
	return Output;
}

TArray<float> UNeuroAIBPLibrary::NeuroActivationFunction_ArgMax(TArray<float> Input)
{
	// This one only gives the index of the highest value in the input array
	TArray<float> Output;
	float Highest = FLT_MIN;
	int32 HighestIndex = 0;
	for(int32 I = 0; I < Input.Num(); I ++)
	{
		if(Input[I] > Highest)
		{
			Highest = Input[I];
			HighestIndex = I;
		}
	}

	while (Output.Num() < Input.Num())
	{
		Output.Add(HighestIndex);
	}
	
	return Output;
}

TArray<float> UNeuroAIBPLibrary::NeuroActivationFunction_SoftMax(TArray<float> Input)
{
	TArray<float> Output;
	TArray<float> Exp = NeuroActivationFunction_Exponential(Input);

	float Sum = 0.f;
	for(float Value : Exp)
	{
		Sum += Value;
	}

	for(int32 I = 0; I < Input.Num(); I ++)
	{
		Output.Add(Exp[I] / Sum);
	}

	return Output;
}

TArray<float> UNeuroAIBPLibrary::NeuroActivationFunction_Exponential(TArray<float> Input)
{
	TArray<float> Output;
	
	for(int32 I = 0; I < Input.Num(); I ++)
	{
		Output.Add( FMath::Pow(EULERS_NUMBER, Input[I]));
	}
	
	return Output;
}

FNeuroLobe UNeuroAIBPLibrary::GenerateRandomNeuroLobe(int32 NumInputs, int32 NumOutputs, int32 NumHiddenLayers,
                                                      int32 HiddenLayerSize, ENeuroActivationFunction InputFunction, ENeuroActivationFunction HLFunction,
                                                      ENeuroActivationFunction OutputFunction)
{
	FNeuroLobe NewLobe = FNeuroLobe();
	
	//Create layers
	for(int32 L = 0; L < NumHiddenLayers + 2; L++)
	{
		const int32 NumNodes = L == 0? NumInputs : (L == NumHiddenLayers + 1? NumOutputs : HiddenLayerSize);
		const ENeuroActivationFunction LayerFunction = L == 0? InputFunction : (L == NumHiddenLayers + 1? OutputFunction : HLFunction);
		FNeuroLayer NewLayer = GenerateRandomNeuroLayer(NumNodes, LayerFunction);
		NewLobe.LobeLayers.Add(NewLayer);

		//Create connections to previous layer
		if(L > 0)
		{
			// For each node in the previous layer
			for(int32 PLN = 0; PLN < NewLobe.LobeLayers[L-1].LayerNodes.Num(); PLN++)
			{
				// Each node in the new layer
				for(int32 ConIndex = 0; ConIndex < NumNodes; ConIndex++)
				{
					// Adds a connection to it
					const int32 ConnectTo = FMath::RandRange(0, NumNodes - 1);
					NewLobe.LobeLayers[L].LayerNodes[ConnectTo].InputWeightMap.Emplace(PLN, FMath::RandRange(-1.f, 1.f));
				}
			}
		}
	}

	for(int32 I = 0; I < NumInputs; I++)
	{
		const FName Name = FName(FString("Input_") + FString::FromInt(I));
		NewLobe.InputNames.Add(Name);
	}

	for(int32 O = 0; O < NumOutputs; O++)
	{
		const FName Name = FName(FString("Output_") + FString::FromInt(O));
		NewLobe.InputNames.Add(Name);
	}

	return NewLobe;
}

FNeuroLayer UNeuroAIBPLibrary::GenerateRandomNeuroLayer(int32 NumNodes, ENeuroActivationFunction ActivationFunction)
{
	FNeuroLayer NewLayer = FNeuroLayer();
	NewLayer.LayerActivationFunction = ActivationFunction;
	
	for(int32 N = 0; N < NumNodes; N++)
	{
		FNeuroNode Node = FNeuroNode();
		Node.Bias = FMath::RandRange(-1.f, 1.f);
		NewLayer.LayerNodes.Add(Node);
	}

	return NewLayer;
}

FNeuroLobe UNeuroAIBPLibrary::MutateLobeSimple(const FNeuroLobe InLobe, int32 NumWeightMutations,
	int32 NumBiasesMutations, int32 MaximumDeltaWeights, int32 MaximumDeltaBiases)
{
	// Catch improper networks
	if(InLobe.LobeLayers.Num() < 2)
	{
		return InLobe;
	}
	
	FNeuroLobe NewLobe = InLobe;

	TArray<TPair<int32, int32>> LayerNodeWeightMutationSites;
	TArray<int32> LayerNodeWeightMutationIndices;
	// Create list of mutation sites for weights
	while(LayerNodeWeightMutationSites.Num() < NumWeightMutations)
	{
		const int32 Layer = FMath::RandRange(0, NewLobe.LobeLayers.Num() - 1);
		const int32 Node = FMath::RandRange(0, NewLobe.LobeLayers[Layer].LayerNodes.Num() - 1);
		const int32 Index = FMath::RandRange(0, NewLobe.LobeLayers[Layer].LayerNodes[Node].InputWeightMap.Num() - 1);

		LayerNodeWeightMutationSites.Add(TPair<int32, int32>(Layer, Node));
		LayerNodeWeightMutationIndices.Add(Index);
	}

	// Mutate the weights
	for(int32 X = 0; X < LayerNodeWeightMutationSites.Num(); X++)
	{
		const TPair<int32, int32> MutationSite = LayerNodeWeightMutationSites[X];
		const int32 MutationIndex = LayerNodeWeightMutationIndices[X];
		TArray<int32> Keys;
		TArray<float> Weights;
		NewLobe.LobeLayers[MutationSite.Key].LayerNodes[MutationSite.Value].InputWeightMap.GenerateKeyArray(Keys);
		NewLobe.LobeLayers[MutationSite.Key].LayerNodes[MutationSite.Value].InputWeightMap.GenerateValueArray(Weights);
		const float NewWeight = Weights[MutationIndex] += FMath::RandRange(-MaximumDeltaWeights, MaximumDeltaWeights);
		NewLobe.LobeLayers[MutationSite.Key].LayerNodes[MutationSite.Value].InputWeightMap.Emplace(Keys[MutationIndex], NewWeight);
	}
	
	TArray<TPair<int32, int32>> LayerNodeBiasMutationSites;
	// Create list of mutation sites for biases
	while(LayerNodeBiasMutationSites.Num() < NumBiasesMutations)
	{
		const int32 Layer = FMath::RandRange(0, NewLobe.LobeLayers.Num() - 1);
		const int32 Node = FMath::RandRange(0, NewLobe.LobeLayers[Layer].LayerNodes.Num() - 1);

		LayerNodeBiasMutationSites.Add(TPair<int32, int32>(Layer, Node));
	}

	// Mutate the biases
	for(const TPair<int32, int32> MutationSite : LayerNodeBiasMutationSites)
	{
		NewLobe.LobeLayers[MutationSite.Key].LayerNodes[MutationSite.Value].Bias += FMath::RandRange(-MaximumDeltaBiases, MaximumDeltaBiases);
	}

	return NewLobe;
}

FNeuroLobe UNeuroAIBPLibrary::MutateLobeNewHiddenLayer(const FNeuroLobe InLobe)
{
	FNeuroLobe NewLobe = InLobe;
	const int32 LayerIndex = NewLobe.LobeLayers.Num() - 2;
	FNeuroLayer NewLayer = NewLobe.LobeLayers[LayerIndex];
	const int32 LayerSize =  NewLayer.LayerNodes.Num();
	
	for(int32 N = 0; N < LayerSize; N++)
	{
		NewLayer.LayerNodes[LayerIndex].InputWeightMap.Empty(LayerSize);
		for(int32 I = 0; I < LayerSize; I++)
		{
			NewLayer.LayerNodes[LayerIndex].InputWeightMap.Emplace(I, N==I? 1.f : 0.f);
		}
	}

	NewLobe.LobeLayers.Insert(NewLayer, LayerIndex + 1);
	return NewLobe;
}

bool UNeuroAIBPLibrary::AreLobesHomologous(const FNeuroLobe A, const FNeuroLobe B)
{
	if(A.LobeLayers.Num() != B.LobeLayers.Num())
	{
		return false;
	}

	for(int32 L = 0; L < A.LobeLayers.Num(); L++)
	{
		if(A.LobeLayers[L].LayerNodes.Num() != B.LobeLayers[L].LayerNodes.Num())
		{
			return false;
		}
		else
		{
			for(int32 N = 0; N < A.LobeLayers[L].LayerNodes.Num(); N++)
			{
				if(A.LobeLayers[L].LayerNodes[N].InputWeightMap.Num() != B.LobeLayers[L].LayerNodes[N].InputWeightMap.Num())
				{
					return false;
				}
				else
				{
					TArray<int32> KeysA;
					TArray<int32> KeysB;
					A.LobeLayers[L].LayerNodes[N].InputWeightMap.GenerateKeyArray(KeysA);
					B.LobeLayers[L].LayerNodes[N].InputWeightMap.GenerateKeyArray(KeysB);
					if(KeysA != KeysB)
					{
						return false;
					}
				}
			}
		}
	}

	return true;
}

FNeuroLobe UNeuroAIBPLibrary::BreedHomologousLobesSimple(const FNeuroLobe A, const FNeuroLobe B)
{
	if(!AreLobesHomologous(A, B))
	{
		return A;
	}

	FNeuroLobe NewLobe = A;

	for(int32 L = 0; L < A.LobeLayers.Num(); L++)
	{
		for(int32 N = 0; N < NewLobe.LobeLayers[L].LayerNodes.Num(); N++)
		{
			// Create a new bias value in between that of the parents.
			const float NewBias = FMath::Lerp(FMath::FRand(), A.LobeLayers[L].LayerNodes[N].Bias, B.LobeLayers[L].LayerNodes[N].Bias);
			NewLobe.LobeLayers[L].LayerNodes[N].Bias = NewBias;
			for(int32 W = 0; W < NewLobe.LobeLayers[L].LayerNodes[N].InputWeightMap.Num(); W++)
			{
				TArray<int32> Keys;
				TArray<float> ValuesA;
				TArray<float> ValuesB;
				NewLobe.LobeLayers[L].LayerNodes[N].InputWeightMap.GenerateKeyArray(Keys);
				B.LobeLayers[L].LayerNodes[N].InputWeightMap.GenerateValueArray(ValuesA);
				A.LobeLayers[L].LayerNodes[N].InputWeightMap.GenerateValueArray(ValuesB);
				// Create a new weight value in between that of the parents.
				const float NewWeight = FMath::Lerp(FMath::FRand(), ValuesA[W], ValuesB[W]);
				NewLobe.LobeLayers[L].LayerNodes[N].InputWeightMap.Emplace(Keys[W], NewWeight);
			}
		}
	}

	// Return the new lobe
	return NewLobe;
}

FNeuroGeneration UNeuroAIBPLibrary::BreedNewGeneration(const FNeuroGeneration InGeneration, int32 NumBreeding,
	int32 OffspringPerPair)
{
	if(NumBreeding > InGeneration.GenerationLobes.Num())
	{
		NumBreeding = InGeneration.GenerationLobes.Num();
	}
	const int32 TotalPairings = (NumBreeding * (NumBreeding - 1)) / 2;

	// -- Begin sorting lobes
	TPriorityQueue<FNeuroLobe> RankedLobes;
	TArray<float> Scores;
	for(TPair<FNeuroLobe, float> Lobe : InGeneration.GenerationLobes)
	{
		Scores.Add(Lobe.Value);
	}
	float MaxScore;
	int32 MaxIndex;
	UKismetMathLibrary::MaxOfFloatArray(Scores, MaxIndex, MaxScore);
	for(TPair<FNeuroLobe, float> Lobe : InGeneration.GenerationLobes)
	{
		RankedLobes.Push(Lobe.Key, MaxScore - Lobe.Value);
	}

	TArray<FNeuroLobe> BreedingLobes;
	// -- Iteration
	for (int32 X = 0; X < TotalPairings; X++)
	{
		FNeuroLobe Current = RankedLobes.Pop(); // Removes it from the top of the heap
		BreedingLobes.Add(Current);
	}

	for(int32 A = 0; A < NumBreeding - 1; A++)
	{
		// 1,2 1,3 1,4 2,3 2,4 3,4
		for(int32 B = A + 1; B < NumBreeding - 1; B++)
		{
			for(int32 C = 0; C < OffspringPerPair; C++)
			{
				FNeuroLobe NewLobe = BreedHomologousLobesSimple(BreedingLobes[A], BreedingLobes[B]);
			}
		}
	}

	FNeuroGeneration NewGeneration = FNeuroGeneration();
	for (int32 X = 0; X < TotalPairings; X++)
	{
		NewGeneration.GenerationLobes.Emplace(BreedingLobes[X], 0.f);
	}

	return NewGeneration;
}

FNeuroGeneration UNeuroAIBPLibrary::MutateGenerationAddInputs(const FNeuroGeneration InGeneration,
	TArray<FName> InputNames)
{
	FNeuroGeneration NewGeneration = InGeneration;
	for(TPair<FNeuroLobe, float>& Lobe: NewGeneration.GenerationLobes)
	{
		Lobe.Key.InputNames.Append(InputNames);
		FNeuroLayer & Layer = Lobe.Key.LobeLayers[0];
		for(FNeuroNode & Node : Layer.LayerNodes)
		{
			int32 InputIndex = Node.InputWeightMap.Num();
			for (int32 N = 0; N < InputNames.Num(); N++)
			{
				Node.InputWeightMap.Emplace((InputIndex + N), 0.f);
			}
		}
	}

	return NewGeneration;
}

FNeuroGeneration UNeuroAIBPLibrary::MutateGenerationRemoveInputs(const FNeuroGeneration InGeneration,
	TArray<int32> InputIndicesToRemove)
{
	FNeuroGeneration NewGeneration = InGeneration;
	for(TPair<FNeuroLobe, float>& Lobe: NewGeneration.GenerationLobes)
	{
		for(int32 Index: InputIndicesToRemove)
		{
			Lobe.Key.InputNames.RemoveAt(Index);
			FNeuroLayer & Layer = Lobe.Key.LobeLayers[0];
			for(FNeuroNode & Node : Layer.LayerNodes)
			{
				Node.InputWeightMap.Remove(Index);
			}
		}
	}

	return NewGeneration;
}

void UNeuroAIBPLibrary::SurviveLobes(const FNeuroGeneration SurviveFrom, FNeuroGeneration& SurviveTo,
	int32 NumToSurvive)
{
	TPriorityQueue<FNeuroLobe> RankedLobes;
	TArray<float> Scores;
	for(TPair<FNeuroLobe, float> Lobe : SurviveFrom.GenerationLobes)
	{
		Scores.Add(Lobe.Value);
	}
	float MaxScore;
	int32 MaxIndex;
	UKismetMathLibrary::MaxOfFloatArray(Scores, MaxIndex, MaxScore);
	for(TPair<FNeuroLobe, float> Lobe : SurviveFrom.GenerationLobes)
	{
		RankedLobes.Push(Lobe.Key, MaxScore - Lobe.Value);
	}

	TArray<TPair<FNeuroLobe, float>> SurvivingLobes;
	// -- Iteration
	for (int32 X = 0; X < NumToSurvive; X++)
	{
		FNeuroLobe Current = RankedLobes.Pop(); // Removes it from the top of the heap
		SurvivingLobes.Emplace(Current, 0.f);
	}

	SurviveTo.GenerationLobes.Append(SurvivingLobes);
}

