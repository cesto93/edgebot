package cmd

import (
	"fmt"
	"strings"

	"edgebot/llm"
)

// lastAssistantText returns the trimmed string content of the final message,
// or an error when the LLM returned no messages or an empty/missing response.
func lastAssistantText(resultMessages []llm.Message) (string, error) {
	if len(resultMessages) == 0 {
		return "", fmt.Errorf("no response from LLM")
	}

	lastMsg := resultMessages[len(resultMessages)-1]
	content, ok := lastMsg.Content.(string)
	if !ok || strings.TrimSpace(content) == "" {
		return "", fmt.Errorf("empty response from LLM")
	}

	return strings.TrimSpace(content), nil
}

// lastAssistantOrLastContent returns the trimmed content of the last
// assistant message, falling back to the last message's content. Bot replies
// use it because they must always send something back, unlike commit and
// structured commands which fail on an empty response.
func lastAssistantOrLastContent(messages []llm.Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "assistant" {
			return contentString(messages[i].Content)
		}
	}
	if len(messages) > 0 {
		return contentString(messages[len(messages)-1].Content)
	}
	return ""
}

func contentString(content any) string {
	if s, ok := content.(string); ok {
		return strings.TrimSpace(s)
	}
	return strings.TrimSpace(fmt.Sprintf("%v", content))
}
