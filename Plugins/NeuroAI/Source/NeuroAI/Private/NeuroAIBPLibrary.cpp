// Copyright Epic Games, Inc. All Rights Reserved.

#include "NeuroAIBPLibrary.h"

#include "EnvironmentQuery/EnvQueryTypes.h"
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

FNeuroLobe UNeuroAIBPLibrary::GenerateRandomNeuroLobe(TArray<FName> InputNames, TArray<FName> OutputNames, int32 NumHiddenLayers,
int32 HiddenLayerSize, ENeuroActivationFunction InputFunction, ENeuroActivationFunction HLFunction, ENeuroActivationFunction OutputFunction)
{
	FNeuroLobe NewLobe = FNeuroLobe();
	int32 PLN = InputNames.Num();
	
	//Create layers
	for(int32 L = 0; L < NumHiddenLayers + 2; L++)
	{
		const int32 NumNodes = L == 0? InputNames.Num() : (L == NumHiddenLayers + 1? OutputNames.Num() : HiddenLayerSize);
		const ENeuroActivationFunction LayerFunction = L == 0? InputFunction : (L == NumHiddenLayers + 1? OutputFunction : HLFunction);
		FNeuroLayer NewLayer = GenerateRandomNeuroLayer(NumNodes, LayerFunction, PLN);
		NewLobe.LobeLayers.Add(NewLayer);
		
		PLN = NumNodes;
	}

	for(int32 I = 0; I < InputNames.Num(); I++)
	{
		//const FName Name = FName(FString("Input_") + FString::FromInt(I));
		NewLobe.InputNames.Add(InputNames[I]);
	}

	for(int32 O = 0; O < OutputNames.Num(); O++)
	{
		//const FName Name = FName(FString("Output_") + FString::FromInt(O));
		NewLobe.OutputNames.Add(OutputNames[O]);
	}

	return NewLobe;
}

FNeuroLayer UNeuroAIBPLibrary::GenerateRandomNeuroLayer(int32 NumNodes, ENeuroActivationFunction ActivationFunction, int32 NumPreviousLayerNodes)
{
	FNeuroLayer NewLayer = FNeuroLayer();
	NewLayer.LayerActivationFunction = ActivationFunction;
	
	for(int32 N = 0; N < NumNodes; N++)
	{
		FNeuroNode Node = FNeuroNode();
		Node.Bias = FMath::RandRange(-1.f, 1.f);
		for(int32 C = 0; C < NumPreviousLayerNodes; C++)
		{
			Node.InputWeightMap.Emplace(C, FMath::RandRange(-1.f, 1.f));
		}
		NewLayer.LayerNodes.Add(Node);
	}

	return NewLayer;
}

FNeuroLobe UNeuroAIBPLibrary::MutateLobeSimple(const FNeuroLobe InLobe, int32 NumWeightMutations,
                                               int32 NumBiasesMutations, float MaximumDeltaWeights, float MaximumDeltaBiases)
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
			const float NewBias = FMath::FRand() >= 0.5f? A.LobeLayers[L].LayerNodes[N].Bias: B.LobeLayers[L].LayerNodes[N].Bias;
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
				const float NewWeight = FMath::FRand() >= 0.5f? ValuesA[W] : ValuesB[W];
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
	//const int32 TotalPairings = (NumBreeding * (NumBreeding - 1)) / 2;

	// -- Begin sorting lobes
	TPriorityQueue<FNeuroLobe> RankedLobes;
	float MaxScore = 0.f;
	for(int32 Score : InGeneration.GenerationScores)
	{
		MaxScore = FMath::Max(MaxScore, Score);
	}

	for(int32 L = 0; L < InGeneration.GenerationLobes.Num(); L++)
	{
		RankedLobes.Push(InGeneration.GenerationLobes[L], MaxScore - InGeneration.GenerationScores[L]);
	}
	

	TArray<FNeuroLobe> BreedingLobes;
	// -- Iteration
	for (int32 X = 0; X < NumBreeding; X++)
	{
		FNeuroLobe Current = RankedLobes.Pop(); // Removes it from the top of the heap
		BreedingLobes.Add(Current);
	}

	TArray<FNeuroLobe> NewLobes;
	for(int32 A = 0; A < NumBreeding - 1; A++)
	{
		// 1,2 1,3 1,4 2,3 2,4 3,4
		for(int32 B = A + 1; B < NumBreeding; B++)
		{
			for(int32 C = 0; C < OffspringPerPair; C++)
			{
				FNeuroLobe NewLobe = BreedHomologousLobesSimple(BreedingLobes[A], BreedingLobes[B]);
				NewLobes.Add(NewLobe);
			}
		}
	}

	FNeuroGeneration NewGeneration = FNeuroGeneration();
	for (int32 X = 0; X < NewLobes.Num(); X++)
	{
		NewGeneration.GenerationLobes.Add(NewLobes[X]);
		NewGeneration.GenerationScores.Add(0.f);
	}

	return NewGeneration;
}

FNeuroGeneration UNeuroAIBPLibrary::MutateGenerationAddInputs(const FNeuroGeneration InGeneration,
	TArray<FName> InputNames)
{
	FNeuroGeneration NewGeneration = InGeneration;
	for(int32 L = 0; L < NewGeneration.GenerationLobes.Num(); L++)
	{
		TPair<FNeuroLobe, float> Lobe;
		Lobe.Key = NewGeneration.GenerationLobes[L];
		Lobe.Value = NewGeneration.GenerationScores[L];
		
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

FNeuroGeneration UNeuroAIBPLibrary::MutateGenerationAddLayer(const FNeuroGeneration InGeneration)
{
	FNeuroGeneration NewGeneration = InGeneration;
	for(int32 L = 0; L < NewGeneration.GenerationLobes.Num(); L++)
	{
		NewGeneration.GenerationLobes[L] = MutateLobeNewHiddenLayer(NewGeneration.GenerationLobes[L]);
	}
	return NewGeneration;
}

FNeuroGeneration UNeuroAIBPLibrary::MutateGenerationRemoveInputs(const FNeuroGeneration InGeneration,
                                                                 TArray<int32> InputIndicesToRemove)
{
	FNeuroGeneration NewGeneration = InGeneration;
	for(int32 L = 0; L < NewGeneration.GenerationLobes.Num(); L++)
	{
		TPair<FNeuroLobe, float> Lobe;
		Lobe.Key = NewGeneration.GenerationLobes[L];
		Lobe.Value = NewGeneration.GenerationScores[L];
		
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

FNeuroGeneration UNeuroAIBPLibrary::MutateGenerationSimple(const FNeuroGeneration InGeneration,
	int32 NumWeightMutations, int32 NumBiasesMutations, float MaximumDeltaWeights, float MaximumDeltaBiases)
{
	FNeuroGeneration NewGeneration = InGeneration;
	for(int32 L = 0; L < NewGeneration.GenerationLobes.Num(); L++)
	{
		NewGeneration.GenerationLobes[L] = MutateLobeSimple(NewGeneration.GenerationLobes[L], NumWeightMutations, NumBiasesMutations, MaximumDeltaWeights, MaximumDeltaBiases);
	}
	return NewGeneration;
}

FNeuroGeneration UNeuroAIBPLibrary::SurviveLobes(const FNeuroGeneration SurviveFrom, const FNeuroGeneration SurviveTo,
                                     int32 NumToSurvive)
{
	FNeuroGeneration NewGeneration = SurviveTo;
	TPriorityQueue<FNeuroLobe> RankedLobes;
	float MaxScore = 0.f;
	for(int32 Score : SurviveFrom.GenerationScores)
	{
		MaxScore = FMath::Max(MaxScore, Score);
	}
	
	for(int32 L = 0; L < SurviveFrom.GenerationLobes.Num(); L++)
	{
		RankedLobes.Push(SurviveFrom.GenerationLobes[L], MaxScore - SurviveFrom.GenerationScores[L]);
	}

	TArray<TPair<FNeuroLobe, float>> SurvivingLobes;
	// -- Iteration
	for (int32 X = 0; X < NumToSurvive; X++)
	{
		FNeuroLobe Current = RankedLobes.Pop(); // Removes it from the top of the heap
		SurvivingLobes.Emplace(Current, 0.f);
	}

	for(TPair<FNeuroLobe, float> Lobe : SurvivingLobes)
	{
		NewGeneration.GenerationLobes.Add(Lobe.Key);
		NewGeneration.GenerationScores.Add(Lobe.Value);
	}

	return NewGeneration;
}

FNeuroLineage UNeuroAIBPLibrary::AppendGenerationToLineage(const FNeuroLineage InLineage,
	const FNeuroGeneration InGeneration)
{
	FNeuroLineage NewLineage = InLineage;
	NewLineage.LineageGenerations.Add(InGeneration);
	return NewLineage;
}

FNeuroGeneration UNeuroAIBPLibrary::SetGenerationScores(const FNeuroGeneration InGeneration,
	const TArray<float> InScores)
{
	FNeuroGeneration NewGeneration = InGeneration;
	NewGeneration.SetGenerationScores(InScores);
	return NewGeneration;
}

FNeuroLineage UNeuroAIBPLibrary::SetLastGenerationScores(const FNeuroLineage InLineage, const TArray<float> InScores)
{
	FNeuroLineage NewLineage = InLineage;
	if(NewLineage.LineageGenerations.Num() > 0)
	{
		NewLineage.LineageGenerations[NewLineage.LineageGenerations.Num() - 1].SetGenerationScores(InScores);
	}
	return NewLineage;
}

FNeuroGeneration UNeuroAIBPLibrary::SetGenerationLobes(const FNeuroGeneration InGeneration,
                                                       const TArray<FNeuroLobe> InLobes)
{
	FNeuroGeneration NewGeneration = InGeneration;
	NewGeneration.SetGenerationLobes(InLobes);
	return NewGeneration;
}

FNeuroGeneration UNeuroAIBPLibrary::GetLatestGeneration(const FNeuroLineage InLineage)
{
	return InLineage.LineageGenerations[InLineage.LineageGenerations.Num() - 1];
}

TArray<FNeuroLobe> UNeuroAIBPLibrary::GetGenerationLobes(const FNeuroGeneration InGeneration)
{
	TArray<FNeuroLobe> OutLobes;
	for(FNeuroLobe Lobe : InGeneration.GenerationLobes)
	{
		OutLobes.Add(Lobe);
	}
	return OutLobes;
}

TArray<float> UNeuroAIBPLibrary::EvaluateLobe(FNeuroLobe InLobe, const TArray<float> Inputs)
{
	return InLobe.FeedForward(Inputs);
}

