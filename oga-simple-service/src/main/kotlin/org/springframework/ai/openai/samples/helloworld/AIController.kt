package org.springframework.ai.openai.samples.helloworld

import org.springframework.ai.chat.client.ChatClient
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import java.util.Map

@RestController
internal class AIController(private val chatClient: ChatClient) {

    @GetMapping("/ai")
    fun completion(
        @RequestParam(
            value = "message",
            defaultValue = "Tell me a joke"
        ) message: String?
    ): MutableMap<String?, String?> {
        return Map.of<String?, String?>(
            "completion",
            chatClient.prompt()
                .user(message)
                .call()
                .content()
        )
    }
}
