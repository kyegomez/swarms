# Enhanced call_llm method for Agent class - Fix for issue #936

def call_llm(
    self,
    task: str,
    img: Optional[str] = None,
    current_loop: int = 0,
    streaming_callback: Optional[Callable[[str], None]] = None,
    *args,
    **kwargs,
) -> str:
    """
    ENHANCED: Calls the appropriate method on the `llm` object based on the given task.
    
    This method has been enhanced to properly handle streaming responses while
    maintaining tool functionality. It accumulates streaming chunks into a 
    complete response before tool processing to prevent JSON parsing errors.

    Args:
        task (str): The task to be performed by the `llm` object.
        img (str, optional): Path or URL to an image file.
        current_loop (int): Current loop iteration for debugging/display.
        streaming_callback (Optional[Callable[[str], None]]): Callback function to receive streaming tokens in real-time.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        str: The complete response from the LLM, properly accumulated from streaming chunks.

    Raises:
        AttributeError: If no suitable method is found in the llm object.
        TypeError: If task is not a string or llm object is None.
        ValueError: If task is empty.
        AgentLLMError: If there's an error calling the LLM.
    """

    # Filter out is_last from kwargs if present
    if "is_last" in kwargs:
        del kwargs["is_last"]

    try:
        # Check if tools are available for logging
        has_tools = exists(self.tools) or exists(self.tools_list_dictionary)
        
        if has_tools and self.verbose:
            logger.debug(f"Agent has tools available, ensuring proper streaming handling for tool parsing")

        # Set streaming parameter in LLM if streaming is enabled
        if self.streaming_on and hasattr(self.llm, "stream"):
            original_stream = self.llm.stream
            self.llm.stream = True

            # Execute LLM call with streaming
            if img is not None:
                streaming_response = self.llm.run(
                    task=task, img=img, *args, **kwargs
                )
            else:
                streaming_response = self.llm.run(
                    task=task, *args, **kwargs
                )

            # Handle streaming response - this is the key fix
            if hasattr(streaming_response, "__iter__") and not isinstance(streaming_response, str):
                
                # Initialize chunk collection
                chunks = []
                
                # Process streaming response chunks
                for chunk in streaming_response:
                    try:
                        # Extract content from chunk (handle different response formats)
                        if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, "content") and delta.content:
                                content = delta.content
                                chunks.append(content)
                                
                                # Call streaming callback if provided (for ConcurrentWorkflow integration)
                                if streaming_callback is not None:
                                    streaming_callback(content)
                                
                                # Display streaming content if print_on is True (but not for callbacks)
                                elif self.print_on and streaming_callback is None:
                                    print(content, end="", flush=True)
                                    
                        elif hasattr(chunk, "content"):
                            content = chunk.content
                            chunks.append(content)
                            
                            if streaming_callback is not None:
                                streaming_callback(content)
                            elif self.print_on and streaming_callback is None:
                                print(content, end="", flush=True)
                                
                        elif isinstance(chunk, str):
                            chunks.append(chunk)
                            
                            if streaming_callback is not None:
                                streaming_callback(chunk)
                            elif self.print_on and streaming_callback is None:
                                print(chunk, end="", flush=True)
                                
                    except Exception as chunk_error:
                        logger.warning(f"Error processing streaming chunk: {chunk_error}")
                        # Continue processing other chunks
                        continue
                
                # Ensure newline after streaming display
                if self.print_on and streaming_callback is None and chunks:
                    print()  # New line after streaming content
                
                # Combine all chunks into complete response
                complete_response = "".join(chunks)
                
                # Log successful streaming accumulation for tool processing
                if has_tools and complete_response:
                    logger.info(f"Successfully accumulated {len(chunks)} streaming chunks into complete response for tool processing")
                
                # Restore original stream setting
                self.llm.stream = original_stream
                
                return complete_response
                
            else:
                # If response is already a string (not streaming), return as-is
                self.llm.stream = original_stream
                return streaming_response
                
        else:
            # Non-streaming call - original behavior
            if img is not None:
                response = self.llm.run(
                    task=task, img=img, *args, **kwargs
                )
            else:
                response = self.llm.run(task=task, *args, **kwargs)

            return response

    except Exception as e:
        # Enhanced error logging with context
        error_context = {
            "task_length": len(task) if task else 0,
            "has_img": img is not None,
            "streaming_on": self.streaming_on,
            "has_tools": exists(self.tools),
            "current_loop": current_loop,
        }
        
        logger.error(
            f"Error calling LLM in Agent '{self.agent_name}': {e}. "
            f"Context: {error_context}"
        )
        
        # Re-raise as AgentLLMError for proper exception handling
        raise AgentLLMError(
            f"LLM call failed for agent '{self.agent_name}': {str(e)}"
        ) from e


def tool_execution_retry(self, response: str, loop_count: int) -> None:
    """
    ENHANCED: Tool execution with proper logging and retry logic.
    
    This method has been enhanced to provide better logging for tool execution
    attempts and results, addressing the missing tool execution logging issue.

    Args:
        response (str): The complete response from the LLM
        loop_count (int): Current loop iteration
    """
    try:
        # Enhanced logging for tool execution attempts
        logger.info(
            f"[Loop {loop_count}] Attempting tool execution from response. "
            f"Response length: {len(response)} chars. "
            f"Available tools: {len(self.tools) if self.tools else 0}"
        )
        
        # Log a preview of the response for debugging
        response_preview = response[:200] + "..." if len(response) > 200 else response
        logger.debug(f"Response preview for tool parsing: {response_preview}")
        
        # Execute tools with retry logic
        for attempt in range(self.tool_retry_attempts):
            try:
                # Call the original tool execution method
                if hasattr(self, 'execute_tools'):
                    tool_result = self.execute_tools(response)
                else:
                    # Fallback to tool_struct execution
                    tool_result = self.tool_struct.execute_function_calls_from_api_response(response)
                
                # Log successful tool execution
                if tool_result:
                    logger.info(
                        f"[Loop {loop_count}] Tool execution successful on attempt {attempt + 1}. "
                        f"Result length: {len(str(tool_result)) if tool_result else 0} chars"
                    )
                    
                    # Add tool result to memory if configured
                    if self.tool_call_summary and tool_result:
                        tool_summary = f"Tool execution result: {str(tool_result)[:500]}..."
                        self.short_memory.add(role="System", content=tool_summary)
                        
                    return  # Successful execution, exit retry loop
                else:
                    logger.debug(f"[Loop {loop_count}] No tool calls detected in response on attempt {attempt + 1}")
                    return  # No tools to execute, normal case
                    
            except Exception as tool_error:
                logger.warning(
                    f"[Loop {loop_count}] Tool execution attempt {attempt + 1} failed: {tool_error}"
                )
                
                if attempt == self.tool_retry_attempts - 1:
                    # Final attempt failed
                    logger.error(
                        f"[Loop {loop_count}] All {self.tool_retry_attempts} tool execution attempts failed. "
                        f"Final error: {tool_error}"
                    )
                    
                    # Add error to memory for context
                    error_msg = f"Tool execution failed after {self.tool_retry_attempts} attempts: {str(tool_error)}"
                    self.short_memory.add(role="System", content=error_msg)
                else:
                    # Wait before retry
                    time.sleep(0.5 * (attempt + 1))  # Progressive backoff
                    
    except Exception as e:
        logger.error(f"[Loop {loop_count}] Critical error in tool execution retry logic: {e}")
        # Don't re-raise to avoid breaking the main execution loop
        

def execute_tools(self, response: str) -> Optional[str]:
    """
    ENHANCED: Execute tools from the LLM response with enhanced error handling and logging.
    
    Args:
        response (str): The complete response from the LLM
        
    Returns:
        Optional[str]: Tool execution result or None if no tools executed
    """
    try:
        if not self.tools:
            return None
            
        # Use the tool_struct to execute tools
        if hasattr(self, 'tool_struct') and self.tool_struct:
            return self.tool_struct.execute_function_calls_from_api_response(response)
        else:
            logger.warning("No tool_struct available for tool execution")
            return None
            
    except Exception as e:
        logger.error(f"Error in execute_tools: {e}")
        raise AgentToolExecutionError(f"Tool execution failed: {str(e)}") from e
