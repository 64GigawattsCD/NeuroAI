#include "NeuroAITypes.h"
#include "NeuroAIBPLibrary.h"

TArray<int32> FNeuroNode::GetInputIndices() const
{
	TArray<int32> Indices;
	InputWeightMap.GenerateKeyArray(Indices);
	return Indices;
}

TArray<float> FNeuroNode::GetWeightsAsArray() const
{
	TArray<float> Weights;
	InputWeightMap.GenerateValueArray(Weights);
	return Weights;
}

float FNeuroNode::FeedForward(TArray<float> InputValues)
{
	int32 InputIndex = 0;
	float Output = 0.f;
	// Sum while applying weights
	for(float W : GetWeightsAsArray())
	{
		Output += (InputValues[InputIndex] * W);
		InputIndex++;
	}
	// Apply bias
	Output += Bias;
	return Output;
}

TArray<float> FNeuroLayer::FeedForward(TArray<float> InputValues)
{
	TArray<float> LayerOutputs;
	// Evaluate each node
	for(FNeuroNode& Node : LayerNodes)
	{
		TArray<float> NodeInputs;
		for(int32 Index : Node.GetInputIndices())
		{
			NodeInputs.Add(InputValues[Index]);
		}
		LayerOutputs.Add(Node.FeedForward(NodeInputs));
	}

	// Apply activation function
	LayerOutputs = UNeuroAIBPLibrary::NeuroActivationFunction(LayerOutputs, LayerActivationFunction);

	return LayerOutputs;
}

FNeuroLobeInputOutput::FNeuroLobeInputOutput(TArray<float> InputValues, TArray<float> OutputValues)
{
	Input = InputValues;
	Output = OutputValues;
}

void FNeuroLobe::SetDesiredOutputs(TArray<float> InOutputs, int32 Index)
{
	if(Index < 0)
	{
		LobeSnapshots[LobeSnapshots.Num() - 1].DesiredOutput = InOutputs;
	}
	else
	{
		if(LobeSnapshots.Num()-1 >= Index)
		{
			LobeSnapshots[Index].DesiredOutput = InOutputs;
		}
		else
		{
			// Something erroneous happened
		}
	}
}

TArray<float> FNeuroLobe::FeedForward(TArray<float> InputValues)
{
	TArray<float> LayerInput = InputValues;
	for(FNeuroLayer& Layer : LobeLayers)
	{
		LayerInput = Layer.FeedForward(LayerInput);
	}

	LobeSnapshots.Add(FNeuroLobeInputOutput(InputValues, LayerInput));
	return LayerInput;
}

void FNeuroLobe::ClearSnapshots()
{
	LobeSnapshots.Empty();
}

void FNeuroGeneration::SetGenerationLobes(const TArray<FNeuroLobe> InLobes)
{
	for(FNeuroLobe Lobe : InLobes)
	{
		GenerationLobes.Add(Lobe);
		GenerationScores.Add(0.f);
	}
}

void FNeuroGeneration::SetGenerationScores(const TArray<float> InScores)
{
	int32 Index = 0;
	while(Index < InScores.Num() && Index < GenerationScores.Num())
	{
		GenerationScores[Index] = InScores[Index];
		Index++;
	}
}

bool FNeuroGeneration::GetHighestScoringLobe(FNeuroLobe & OutLobe)
{
	int32 MaxIndex = -1;
	float MaxScore = 0.f;
	for(int32 I = 0; I < GenerationScores.Num(); I++)
	{
		const float & Score = GenerationScores[I];
		if(Score > MaxScore)
		{
			MaxScore = Score;
			MaxIndex = I;
		}
	}

	if(MaxIndex > -1)
	{
		OutLobe = GenerationLobes[MaxIndex];
		return true;
	}
	else
	{
		return false;
	}
}

void FNeuroLineage::AppendGeneration(FNeuroGeneration& InGeneration)
{
	LineageGenerations.Add(InGeneration);
}

void FNeuroLineage::AppendGenerations(TArray<FNeuroGeneration> InGenerations)
{
	for(FNeuroGeneration& InGeneration : InGenerations)
	{
		AppendGeneration(InGeneration);
	}
}

FNeuroGeneration FNeuroLineage::GetLatestGeneration()
{
	if(LineageGenerations.Num() > 0)
	{
		return LineageGenerations[LineageGenerations.Num() - 1];
	}
	else
	{
		FNeuroGeneration NewGen = FNeuroGeneration();
		return NewGen;
	}
}
