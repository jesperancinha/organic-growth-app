import dev.langchain4j.memory.chat.MessageWindowChatMemory
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4_O_MINI
import dev.langchain4j.service.AiServices

interface Assistant {
    fun chat(message: String): String
}

fun main() {
    val OPENAI_API_KEY = System.getenv("OPENAI_API_KEY") ?: "demo"

    val model: ChatModel = OpenAiChatModel.builder()
        .apiKey(OPENAI_API_KEY)
        .modelName(GPT_4_O_MINI)
        .build()

    val memory = MessageWindowChatMemory.withMaxMessages(50)
    val assistant = AiServices.builder(Assistant::class.java)
        .chatModel(model)
        .chatMemory(memory)
        .build()

    println(assistant.chat("Hello, my name is Joao"))
    println(assistant.chat("What is my name?"))
}

