package org.springframework.ai.openai.samples.helloworld

import org.springframework.ai.chat.client.ChatClient
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

@Configuration
internal class Config {
    @Bean
    fun chatClient(builder: ChatClient.Builder): ChatClient? {
        return builder.build()
    }
}
