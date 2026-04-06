# LangChain Chain Types Tutorial (Modern LCEL - 2026)
# Covers: Simple Chain, Sequential Chain, Multi-step Pipeline
# No legacy classes: LLMChain / SequentialChain / SimpleSequentialChain removed

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=500
)

print("🚀 LangChain Chain Types Tutorial (Modern LCEL)")
print("=" * 50)

# ============================================================================
# 1. SIMPLE CHAIN (Prompt → LLM → Output)
# ============================================================================
print("\n1️⃣ SIMPLE CHAIN")
print("-" * 20)

story_prompt = ChatPromptTemplate.from_template(
    "Write a short creative story (2-3 sentences) about {topic}. Make it interesting and engaging."
)

# LCEL open pipe syntax: prompt | llm | parser
simple_chain = story_prompt | llm | StrOutputParser()

topic = "a robot learning to paint"
story_result = simple_chain.invoke({"topic": topic})
print(f"Topic: {topic}")
print(f"Generated Story: {story_result}")

# ============================================================================
# 2. Simple SEQUENTIAL CHAIN (Chain output feeds into next chain)
# ============================================================================
print("\n2️⃣ SEQUENTIAL CHAIN")
print("-" * 25)

# Chain 1: Generate a business idea
idea_prompt = ChatPromptTemplate.from_template(
    "Generate a creative business idea for the industry: {industry}. "
    "Provide just the business idea in one sentence."  
)
idea_chain = idea_prompt | llm | StrOutputParser()

# Chain 2: Create a marketing slogan
slogan_prompt = ChatPromptTemplate.from_template(
    "Create a catchy marketing slogan for this business idea: {business_idea}. "
    "Make it memorable and under 10 words."
)
slogan_chain = slogan_prompt | llm | StrOutputParser()


'''When idea_chain runs, it returns a plain string — something like "A subscription box for eco-friendly office supplies".
But slogan_chain expects a dict with the key business_idea, not a raw string.
So the lambda is just a converter in the middle:'''

"""In the old code, SimpleSequentialChain handled this automatically under the hood."""

# Wire them together: output of idea_chain becomes {business_idea} for slogan_chain
sequential_chain = (
    idea_chain
    | (lambda business_idea: {"business_idea": business_idea})
    | slogan_chain
)

'''
# Alternate Way 

def wrap_as_business_idea(text):
    return {"business_idea": text}

sequential_chain = idea_chain | wrap_as_business_idea | slogan_chain

'''

industry = "sustainable technology"
final_result = sequential_chain.invoke({"industry": industry})
print(f"Industry: {industry}")
print(f"Final Marketing Slogan: {final_result}")

# ============================================================================
# 3. SEQUENTIAL CHAIN (MULTI-STEP CHAIN WITH NAMED OUTPUTS)
# ============================================================================
print("\n3️⃣ MULTI-STEP CHAIN (Named Outputs)")
print("-" * 35)

# Chain 1: Market analysis
analysis_prompt = ChatPromptTemplate.from_template(
    """
    Analyze this product concept:
    Product: {product_name}
    Target Market: {target_market}

    Provide a brief market analysis (2-3 sentences).
    """
)
analysis_chain = analysis_prompt | llm | StrOutputParser()

# Chain 2: Pricing strategy (uses product_name + market_analysis)
pricing_prompt = ChatPromptTemplate.from_template(
    """
    Based on this market analysis: {market_analysis}

    Suggest a pricing strategy for {product_name}.
    Include price range and reasoning (2-3 sentences).
    """
)
output_key = "pricing_strategy"  # This will be the key for the output of this chain
pricing_chain = pricing_prompt | llm | output_key | StrOutputParser()

# Chain 3: Business plan summary
business_plan_prompt = ChatPromptTemplate.from_template(
    """
    Create a concise business plan summary using:

    Product: {product_name}
    Target Market: {target_market}
    Market Analysis: {market_analysis}
    Pricing Strategy: {pricing_strategy}

    Summarize in 3-4 sentences focusing on key opportunities.
    """
)
business_plan_chain = business_plan_prompt | llm | StrOutputParser()


def run_pipeline(product_name, target_market):
    market_analysis = analysis_chain.invoke({
        "product_name": product_name,
        "target_market": target_market
    })

    pricing_strategy = pricing_chain.invoke({
        "product_name": product_name,
        "market_analysis": market_analysis
    })

    business_plan = business_plan_chain.invoke({
        "product_name": product_name,
        "target_market": target_market,
        "market_analysis": market_analysis,
        "pricing_strategy": pricing_strategy
    })

    return {
        "market_analysis": market_analysis,
        "pricing_strategy": pricing_strategy,
        "business_plan": business_plan
    }

result = run_pipeline("Smart Fitness Mirror", "health-conscious millennials")

print(result["market_analysis"])
print(result["pricing_strategy"])
print(result["business_plan"])

print("📊 MULTI-STEP CHAIN RESULTS:")
print(f"Product: {inputs['product_name']}")
print(f"Target Market: {inputs['target_market']}")
print(f"\n📈 Market Analysis:\n{market_analysis}")
print(f"\n💰 Pricing Strategy:\n{pricing_strategy}")
print(f"\n📋 Business Plan:\n{business_plan}")


"""
Scenario Approach: 

- One variable flowing throughCombine with | and invoke once #simple seq chain

- Multiple variables at each stepRun separately, pass state manually #seq chains

"""
# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("📚 SUMMARY OF MODERN LCEL CHAIN PATTERNS")
print("=" * 60)

print("""
1️⃣ SIMPLE CHAIN:
   • prompt | llm | StrOutputParser()
   • Single input → single output
   • Best for: one-shot tasks

2️⃣ SEQUENTIAL CHAIN:
   • chain1 | (lambda x: {"key": x}) | chain2
   • Output of one chain becomes input of the next
   • Best for: linear two-step flows

3️⃣ MULTI-STEP CHAIN:
   • Run chains separately, pass state forward manually
   • Full control over inputs/outputs at each step
   • Best for: complex pipelines with multiple named variables
""")

# ============================================================================
# PRACTICAL EXAMPLE: Blog Post Creation Pipeline
# ============================================================================
print("\n🎯 PRACTICAL EXAMPLE: Blog Post Creation Pipeline")
print("-" * 50)

topic_generator = (
    ChatPromptTemplate.from_template(
        "Generate 3 engaging blog post topics about {subject}. List them numbered 1-3."
    )
    | llm
    | StrOutputParser()
)

outline_generator = (
    ChatPromptTemplate.from_template(
        "From these topics: {topics}\n\nPick the first topic and create a detailed outline with 4 main points."
    )
    | llm
    | StrOutputParser()
)

intro_writer = (
    ChatPromptTemplate.from_template(
        "Based on this outline: {outline}\n\nWrite an engaging introduction paragraph (3-4 sentences)."
    )
    | llm
    | StrOutputParser()
)

# Run pipeline step by step
subject = "artificial intelligence in healthcare"
topics = topic_generator.invoke({"subject": subject})
outline = outline_generator.invoke({"topics": topics})
introduction = intro_writer.invoke({"outline": outline})

print(f"Subject: {subject}")
print(f"\nGenerated Topics:\n{topics}")
print(f"\nOutline:\n{outline}")
print(f"\nFinal Blog Introduction:\n{introduction}")

print("\n✅ Tutorial Complete! All chains use modern LCEL — no deprecated classes.")
