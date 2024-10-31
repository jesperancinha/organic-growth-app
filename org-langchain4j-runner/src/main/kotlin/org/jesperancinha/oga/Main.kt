import dev.langchain4j.data.message.UserMessage.userMessage
import dev.langchain4j.internal.Utils.getOrDefault
import dev.langchain4j.memory.ChatMemory
import dev.langchain4j.memory.chat.TokenWindowChatMemory
import dev.langchain4j.model.chat.ChatLanguageModel
import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4_O_MINI
import dev.langchain4j.model.openai.OpenAiTokenizer

val OPENAI_API_KEY: String = getOrDefault(System.getenv("OPENAI_API_KEY"), "demo")

fun main() {
    val chatMemory: ChatMemory = TokenWindowChatMemory.withMaxTokens(300, OpenAiTokenizer())

    val model: ChatLanguageModel = OpenAiChatModel.builder()
        .apiKey(OPENAI_API_KEY)
        .modelName(GPT_4_O_MINI)
        .build()

    chatMemory.add(userMessage("Hello, my name is Joao"))
    val answer = model.generate(chatMemory.messages()).content()
    println(answer.text())
    chatMemory.add(answer)

    chatMemory.add(userMessage("What is my name?"))
    val answerWithName = model.generate(chatMemory.messages()).content()
    println(answerWithName.text())
    chatMemory.add(answerWithName)
}
