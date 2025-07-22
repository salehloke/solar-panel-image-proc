import { Injectable } from '@nestjs/common';
import { CDHService } from 'libs/shared/service/cdh.service';

@Injectable()
export class ExampleTrackingService {
  constructor(private readonly cdhService: CDHService) {}

  /**
   * Example 1: Tracking a user login event
   */
  async trackUserLogin(userId: string, loginMethod: string, success: boolean) {
    this.cdhService.recordTrackingEvent(
      'UserLogin',
      userId,
      {
        eventType: 'login',
        loginMethod,
        success,
        timestamp: new Date(),
        metadata: {
          userAgent: 'web-browser',
          ipAddress: '192.168.1.1',
        },
      }
    );
  }

  /**
   * Example 2: Tracking a data update operation
   */
  async trackDataUpdate(
    userId: string, 
    operationType: string, 
    updatedFields: string[], 
    executionTime: number
  ) {
    this.cdhService.recordTrackingEvent(
      'DataUpdate',
      userId,
      {
        operation: 'update',
        operationType,
        updatedFields,
        executionTimeMs: executionTime,
        timestamp: new Date(),
        details: {
          fieldsCount: updatedFields.length,
          operationStatus: 'success',
        },
      }
    );
  }

  /**
   * Example 3: Tracking an error event
   */
  async trackErrorEvent(
    userId: string, 
    error: Error, 
    context: string, 
    additionalData?: any
  ) {
    this.cdhService.recordTrackingEvent(
      'ErrorEvent',
      userId,
      {
        eventType: 'error',
        errorMessage: error.message,
        errorStack: error.stack,
        context,
        timestamp: new Date(),
        ...additionalData,
      }
    );
  }

  /**
   * Example 4: Tracking a complex operation with multiple steps
   */
  async trackComplexOperation(userId: string, operationName: string) {
    const startTime = new Date();
    const trackerLogs = [];

    try {
      // Step 1: Initialize operation
      trackerLogs.push({
        step: 'Initialization',
        status: 'success',
        data: { userId, operationName },
      });

      // Step 2: Process data
      trackerLogs.push({
        step: 'DataProcessing',
        status: 'in_progress',
        data: { userId, recordsProcessed: 0 },
      });

      // Simulate some processing
      await this.simulateProcessing();

      trackerLogs.push({
        step: 'DataProcessing',
        status: 'success',
        data: { userId, recordsProcessed: 150 },
      });

      // Step 3: Finalize
      trackerLogs.push({
        step: 'Finalization',
        status: 'success',
        data: { userId, finalStatus: 'completed' },
      });

      const endTime = new Date();
      const executionTime = endTime.getTime() - startTime.getTime();

      // Record the complete operation
      this.cdhService.recordTrackingEvent(
        'ComplexOperation',
        userId,
        {
          operation: operationName,
          trackerLogs,
          executionTimeMs: executionTime,
          status: 'success',
          timestamp: new Date(),
        }
      );

    } catch (error) {
      const endTime = new Date();
      const executionTime = endTime.getTime() - startTime.getTime();

      trackerLogs.push({
        step: 'OperationError',
        status: 'failed',
        error: {
          message: error.message,
          code: error.status || 500,
        },
        data: { userId, executionTimeMs: executionTime },
      });

      // Record the failed operation
      this.cdhService.recordTrackingEvent(
        'ComplexOperationError',
        userId,
        {
          operation: operationName,
          trackerLogs,
          executionTimeMs: executionTime,
          status: 'failed',
          error: {
            message: error.message,
            stack: process.env.NODE_ENV === 'production' ? undefined : error.stack,
          },
          timestamp: new Date(),
        }
      );

      throw error;
    }
  }

  /**
   * Example 5: Tracking with API response data
   */
  async trackApiCall(userId: string, apiEndpoint: string, response: any) {
    this.cdhService.recordTrackingEvent(
      'ApiCall',
      userId,
      {
        apiEndpoint,
        requestMethod: 'POST',
        timestamp: new Date(),
        metadata: {
          responseStatus: response.status,
          responseTime: response.responseTime,
        },
      },
      response // Pass the response as the fourth parameter
    );
  }

  /**
   * Example 6: Tracking user action events
   */
  async trackUserAction(
    userId: string, 
    action: string, 
    target: string, 
    success: boolean
  ) {
    this.cdhService.recordTrackingEvent(
      'UserAction',
      userId,
      {
        action,
        target,
        success,
        timestamp: new Date(),
        userContext: {
          sessionId: 'session-123',
          pageUrl: '/dashboard',
        },
      }
    );
  }

  private async simulateProcessing(): Promise<void> {
    // Simulate some async processing
    await new Promise(resolve => setTimeout(resolve, 100));
  }
}

// Usage examples in a controller or service:

/*
// In your controller or service:

@Post('login')
async login(@Body() loginDto: LoginDto) {
  try {
    const result = await this.authService.login(loginDto);
    
    // Track successful login
    this.exampleTrackingService.trackUserLogin(
      result.userId, 
      'email', 
      true
    );
    
    return result;
  } catch (error) {
    // Track failed login
    this.exampleTrackingService.trackUserLogin(
      loginDto.email, 
      'email', 
      false
    );
    
    throw error;
  }
}

@Put('user/:id')
async updateUser(@Param('id') userId: string, @Body() updateDto: UpdateUserDto) {
  const startTime = new Date();
  
  try {
    const result = await this.userService.updateUser(userId, updateDto);
    
    const endTime = new Date();
    const executionTime = endTime.getTime() - startTime.getTime();
    
    // Track the update operation
    this.exampleTrackingService.trackDataUpdate(
      userId,
      'userProfile',
      Object.keys(updateDto),
      executionTime
    );
    
    return result;
  } catch (error) {
    // Track the error
    this.exampleTrackingService.trackErrorEvent(
      userId,
      error,
      'userUpdate',
      { updateDto }
    );
    
    throw error;
  }
}
*/ 